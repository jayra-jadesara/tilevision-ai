"""Customer launches the app and waits until everything is ready."""

from __future__ import annotations

import pytest

from qa_e2e.framework.readiness import probe_readiness

pytestmark = pytest.mark.qa_e2e


def test_ui_is_visible_like_a_customer_launch(session, driver):
    action = session.artifacts.begin("Launch app / UI ready")
    session.human.think(0.4, 1.0)
    driver.goto_dashboard()
    session.human.think(0.3, 0.7)
    driver.goto_search()
    ok = session.main_window.isVisible()
    session.artifacts.end(
        action,
        ok=ok,
        detail="MainWindow visible",
        screenshot_widget=session.main_window,
    )
    assert ok


def test_model_faiss_sqlite_ready(session):
    action = session.artifacts.begin("Verify model + FAISS + SQLite")
    report = probe_readiness(session, require_catalog=False)
    session.artifacts.end(
        action,
        ok=report.ok,
        detail=f"device={report.device} backend={report.faiss_backend}",
        screenshot_widget=session.main_window,
        metrics={
            "model_loaded": report.model_loaded,
            "faiss_loaded": report.faiss_loaded,
            "sqlite_connected": report.sqlite_connected,
            "faiss_count": report.faiss_count,
            "sqlite_count": report.sqlite_count,
            "failures": report.failures,
        },
    )
    assert report.model_loaded, report.failures
    assert report.faiss_loaded, report.failures
    assert report.sqlite_connected, report.failures
    assert report.faiss_backend == "flat_ip"
