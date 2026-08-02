"""Shared post-scenario checks for release validation."""

from __future__ import annotations

from typing import Any, Dict

from PySide6.QtWidgets import QApplication

from qa_e2e.framework.failures import detect_after_search, detect_ui_freeze
from qa_e2e.framework.ui_driver import UIDriver
from src.presentation.viewmodels.search_viewmodel import SearchState


def common_ui_checks(session, *, expect_results: bool = False) -> Dict[str, bool]:
    driver = UIDriver(session)
    freeze = detect_ui_freeze(session)
    state = session.search_viewmodel.state
    checks = {
        "window_visible": bool(session.main_window.isVisible()),
        "no_ui_freeze": freeze.ok,
        "not_searching_forever": state != SearchState.SEARCHING,
        "app_process_events": True,
    }
    QApplication.processEvents()
    if expect_results:
        scan = detect_after_search(session, since=0.0, expect_results=True)
        checks["search_ok"] = scan.ok and state == SearchState.RESULTS
        checks["results_visible"] = driver.result_count_ui() > 0
        table = driver.results_table()
        thumb_ok = False
        for row in range(min(table.rowCount(), 3)):
            item = table.item(row, 0)
            if item is not None and not item.icon().isNull():
                thumb_ok = True
                break
        checks["thumbnail_rendered"] = thumb_ok or driver.result_count_ui() == 0
        # Metadata columns: product/brand
        meta_ok = False
        if table.rowCount() > 0:
            code = table.item(0, 2)
            meta_ok = code is not None and bool(code.text().strip())
        checks["metadata_loaded"] = meta_ok
    return checks


def all_checks_passed(checks: Dict[str, bool]) -> bool:
    return all(checks.values()) if checks else False


def summarize_checks(checks: Dict[str, Any]) -> str:
    bad = [k for k, v in checks.items() if not v]
    return "ok" if not bad else "failed: " + ", ".join(bad)
