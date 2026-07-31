"""Detect customer-visible failures during a live session."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

from PySide6.QtWidgets import QApplication, QProgressBar

from qa_e2e.framework.session import AppSession
from qa_e2e.framework.ui_driver import UIDriver
from src.presentation.viewmodels.search_viewmodel import SearchState


@dataclass
class FailureFinding:
    code: str
    message: str
    fatal: bool = True


@dataclass
class FailureScan:
    findings: List[FailureFinding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(f.fatal for f in self.findings)

    def add(self, code: str, message: str, *, fatal: bool = True) -> None:
        self.findings.append(FailureFinding(code=code, message=message, fatal=fatal))


def detect_after_search(session: AppSession, *, since: float, expect_results: bool = True) -> FailureScan:
    scan = FailureScan()
    driver = UIDriver(session)
    state = session.search_viewmodel.state
    worker = getattr(session.search_viewmodel, "_worker", None)

    if state == SearchState.SEARCHING:
        scan.add("SEARCH_STUCK", "Search still in SEARCHING state after wait")
    if state == SearchState.ERROR:
        scan.add("SEARCH_ERROR", f"Search error UI: {driver.search_status()}")
    if state == SearchState.IDLE and expect_results:
        # Idle without ever searching
        if not session.logs.contains("[SEARCH]", since=since):
            scan.add("SEARCH_NEVER_STARTED", "No [SEARCH] log breadcrumbs after customer drop")

    if worker is not None and not worker.isRunning() and state == SearchState.SEARCHING:
        scan.add("WORKER_EXITED", "SearchWorker exited while UI still shows searching")

    if expect_results and state == SearchState.NO_RESULTS:
        scan.add("NO_RESULTS", "UI shows no results for a query that should match the catalog")

    if expect_results and state == SearchState.RESULTS:
        if driver.result_count_ui() <= 0:
            scan.add("TABLE_EMPTY", "SearchState=results but ResultsTable has 0 rows")
        # Thumbnail column icons
        table = driver.results_table()
        missing_thumbs = 0
        for row in range(min(table.rowCount(), 5)):
            item = table.item(row, 0)
            if item is None or item.icon().isNull():
                missing_thumbs += 1
        if missing_thumbs == min(table.rowCount(), 5) and table.rowCount() > 0:
            scan.add("THUMBNAILS_MISSING", "No thumbnail icons rendered in results", fatal=False)

    # Spinner / progress bar should not stay active forever after settle
    bar: QProgressBar = driver.progress_bar()
    if state != SearchState.SEARCHING and bar.isVisible() and bar.maximum() == 0:
        # indeterminate busy bar still visible
        scan.add("SPINNER_STUCK", "Search progress bar still busy after settle", fatal=False)

    # FAISS / SQLite emptiness
    if session.vector_index.get_total_count() <= 0 and expect_results:
        scan.add("FAISS_EMPTY", "FAISS catalog is empty — search cannot succeed")
    try:
        tiles = session.image_repository.get_all()
        if not tiles and expect_results:
            scan.add("SQLITE_EMPTY", "SQLite tiles table is empty")
    except Exception as exc:
        scan.add("SQLITE_READ_FAIL", str(exc))

    return scan


def detect_ui_freeze(session: AppSession, *, sample_ms: int = 800) -> FailureScan:
    """
    Detect a frozen UI by checking that the event loop still advances.

    If processEvents + a short wait never lets timers/workers progress and
    the window stops accepting events, flag FREEZE.
    """
    scan = FailureScan()
    before = time.monotonic()
    QApplication.processEvents()
    from PySide6.QtTest import QTest

    QTest.qWait(sample_ms)
    QApplication.processEvents()
    after = time.monotonic()
    if after - before < (sample_ms / 1000.0) * 0.5:
        scan.add("UI_FREEZE", "Event loop did not advance for the sample window")
    if not session.main_window.isVisible():
        scan.add("WINDOW_HIDDEN", "Main window is no longer visible")
    return scan
