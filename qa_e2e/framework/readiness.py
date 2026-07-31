"""Wait for customer-visible readiness: model, FAISS, SQLite, catalog."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

from PySide6.QtWidgets import QApplication

from qa_e2e.framework.session import AppSession
from src.utils import search_stages


@dataclass
class ReadinessReport:
    ui_ready: bool = False
    model_loaded: bool = False
    faiss_loaded: bool = False
    sqlite_connected: bool = False
    catalog_indexed: bool = False
    faiss_count: int = 0
    sqlite_count: int = 0
    faiss_backend: str = ""
    device: str = ""
    failures: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = []

    @property
    def ok(self) -> bool:
        return (
            self.ui_ready
            and self.model_loaded
            and self.faiss_loaded
            and self.sqlite_connected
            and not self.failures
        )


def probe_readiness(session: AppSession, *, require_catalog: bool = False) -> ReadinessReport:
    report = ReadinessReport()
    QApplication.processEvents()

    report.ui_ready = bool(session.main_window.isVisible())
    if not report.ui_ready:
        report.failures.append("Main window not visible")

    # DINOv2
    try:
        report.model_loaded = getattr(session.embedder, "_model", None) is not None
        if not report.model_loaded:
            report.failures.append("DINOv2 model is not loaded in memory")
    except Exception as exc:
        report.failures.append(f"Model probe failed: {exc}")

    # FAISS
    try:
        idx = session.vector_index
        report.faiss_count = int(idx.get_total_count())
        report.faiss_backend = str(idx.active_backend().value)
        report.faiss_loaded = getattr(idx, "_index", None) is not None
        if not report.faiss_loaded:
            report.failures.append("FAISS index object is None after load_index()")
        if report.faiss_backend != "flat_ip":
            report.failures.append(f"Expected IndexFlatIP backend flat_ip, got {report.faiss_backend}")
    except Exception as exc:
        report.failures.append(f"FAISS probe failed: {exc}")

    # SQLite
    try:
        with session.db_context.session() as conn:
            conn.execute("SELECT 1")
            row = conn.execute("SELECT COUNT(*) AS c FROM tiles").fetchone()
            report.sqlite_count = int(row[0] if not hasattr(row, "keys") else row["c"])
        report.sqlite_connected = True
    except Exception as exc:
        report.failures.append(f"SQLite probe failed: {exc}")

    report.catalog_indexed = report.faiss_count > 0 and report.sqlite_count > 0
    if require_catalog and not report.catalog_indexed:
        report.failures.append(
            f"Catalog not indexed (faiss={report.faiss_count}, sqlite={report.sqlite_count})"
        )

    report.device = session.embedder.runtime_info.summary_for_ui()
    return report


def wait_for_search_pipeline(
    session: AppSession,
    *,
    since: float,
    timeout: float = 300.0,
    require_stages: bool = True,
) -> List[str]:
    """
    Wait until search settles and return missing pipeline stages (if any).

    Stages are observed from live log breadcrumbs — not simulated.
    """
    from src.presentation.viewmodels.search_viewmodel import SearchState
    from qa_e2e.framework.ui_driver import UIDriver

    driver = UIDriver(session)
    state = driver.wait_search_settled(timeout=timeout)
    expected = [
        search_stages.STAGE_EMBEDDING_GENERATED,
        search_stages.STAGE_EMBEDDING_NORMALIZED,
        search_stages.STAGE_FAISS_SEARCH,
        search_stages.STAGE_SQLITE_HYDRATE,
        search_stages.STAGE_RERANK_COMPLETE,
        search_stages.STAGE_RESULTS_READY,
    ]
    # Cache hit may skip embedding generated — accept either path
    missing = session.logs.wait_for_all(expected, timeout=2.0, since=since)
    # If cache hit, embedding generated may be absent
    if search_stages.STAGE_EMBEDDING_GENERATED in missing:
        if session.logs.contains(search_stages.STAGE_EMBEDDING_CACHE_HIT, since=since):
            missing = [m for m in missing if m != search_stages.STAGE_EMBEDDING_GENERATED]
    if not require_stages:
        return missing
    if state == SearchState.ERROR:
        missing.append("SearchState=ERROR")
    return missing


def wait_until(predicate, *, timeout: float = 60.0, poll: float = 0.2, message: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return
        time.sleep(poll)
    raise TimeoutError(f"Timed out waiting for {message}")
