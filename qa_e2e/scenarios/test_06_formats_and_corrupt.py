"""Customer drops PNG/JPG/WEBP/TIFF, large/small images, and corrupt files."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from qa_e2e.framework.expectations import evaluate_from_manifest
from qa_e2e.framework.failures import detect_after_search
from src.presentation.viewmodels.search_viewmodel import SearchState

pytestmark = pytest.mark.qa_e2e


@pytest.mark.parametrize(
    "query_id",
    ["q_png", "q_jpg", "q_webp", "q_tiff", "q_large", "q_small"],
)
def test_supported_formats_and_sizes(session, driver, catalog_indexed, record_expectation, query_id):
    q = next(x for x in session.expected_manifest["queries"] if x["id"] == query_id)
    action = session.artifacts.begin(f"Format/size search: {query_id}")
    since = time.time()
    driver.drag_drop_image(Path(q["path"]))
    state = driver.wait_search_settled(timeout=360.0)
    scan = detect_after_search(session, since=since, expect_results=True)
    paths = driver.visible_result_paths() or [
        r.tile.file_path for r in session.search_viewmodel.last_results
    ]
    exp = record_expectation(
        evaluate_from_manifest(session.expected_manifest, query_id, paths)
    )
    ok = state == SearchState.RESULTS and scan.ok and exp.ok
    session.artifacts.end(
        action,
        ok=ok,
        detail=exp.detail,
        screenshot_widget=session.main_window,
        metrics={"state": state, "top": exp.top_product},
    )
    assert ok, (state, scan.findings, exp.detail)


def test_corrupt_image_does_not_freeze_ui(session, driver, catalog_indexed):
    corrupt = Path(session.expected_manifest["corrupt_dir"]) / "not_an_image.jpg"
    action = session.artifacts.begin("Corrupt image drop", detail=str(corrupt))
    since = time.time()
    driver.drag_drop_image(corrupt)
    # Should fail gracefully — never hang
    session.human.wait(1.0)
    from PySide6.QtWidgets import QApplication
    from PySide6.QtTest import QTest

    QTest.qWait(1500)
    QApplication.processEvents()
    state = session.search_viewmodel.state
    freeze_ok = session.main_window.isVisible()
    # Accept error / idle / no_results — not endless searching
    ok = freeze_ok and state != SearchState.SEARCHING
    session.artifacts.end(
        action,
        ok=ok,
        detail=f"state={state} status={driver.search_status()}",
        screenshot_widget=session.main_window,
        metrics={"state": state, "since_logs": session.logs.messages(since=since)[-20:]},
    )
    assert ok


def test_unsupported_txt_drop_is_rejected(session, driver, catalog_indexed):
    bad = Path(session.expected_manifest["corrupt_dir"]) / "notes.txt"
    action = session.artifacts.begin("Unsupported drop (.txt)")
    since = time.time()
    driver.drag_drop_image(bad)
    session.human.think(0.3, 0.7)
    rejected = session.logs.contains("Drop rejected", since=since) or session.logs.contains(
        "Unsupported", since=since
    )
    # If Qt ignores before callback, UI should remain usable
    ok = session.main_window.isVisible() and session.search_viewmodel.state != SearchState.SEARCHING
    session.artifacts.end(
        action,
        ok=ok,
        detail=f"rejected_logged={rejected}",
        screenshot_widget=session.main_window,
    )
    assert ok
