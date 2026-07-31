"""Customer cancels a search, then searches again successfully."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from qa_e2e.framework.expectations import evaluate_from_manifest
from qa_e2e.framework.failures import detect_after_search
from src.presentation.viewmodels.search_viewmodel import SearchState

pytestmark = pytest.mark.qa_e2e


def test_cancel_search_then_search_again(session, driver, catalog_indexed, record_expectation):
    queries = [q for q in session.expected_manifest["queries"] if q["kind"] == "crop_match"]
    first, second = queries[0], queries[1]

    action = session.artifacts.begin("Cancel Search", detail=first["id"])
    driver.drag_drop_image(Path(first["path"]))
    # Interrupt quickly like an impatient customer
    session.human.wait(0.15)
    try:
        driver.clear_search()
    except AssertionError:
        # Clear may be disabled until state flips — wait briefly
        session.human.wait(0.4)
        driver.clear_search()
    session.human.think(0.3, 0.8)
    state = session.search_viewmodel.state
    # After clear: idle / cancelled path
    cancelled_ok = state in (SearchState.IDLE, SearchState.ERROR, SearchState.NO_RESULTS, SearchState.RESULTS)
    session.artifacts.end(
        action,
        ok=cancelled_ok,
        detail=f"state_after_clear={state}",
        screenshot_widget=session.main_window,
    )
    assert cancelled_ok

    action2 = session.artifacts.begin("Search after Cancel", detail=second["id"])
    since = time.time()
    driver.drag_drop_image(Path(second["path"]))
    settled = driver.wait_search_settled(timeout=300.0)
    scan = detect_after_search(session, since=since, expect_results=True)
    paths = driver.visible_result_paths() or [
        r.tile.file_path for r in session.search_viewmodel.last_results
    ]
    exp = record_expectation(
        evaluate_from_manifest(session.expected_manifest, second["id"], paths)
    )
    ok = settled == SearchState.RESULTS and scan.ok and exp.ok
    session.artifacts.end(
        action2,
        ok=ok,
        detail=exp.detail,
        screenshot_widget=session.main_window,
        metrics={"state": settled, "results": driver.result_count_ui()},
    )
    assert ok, (scan.findings, exp.detail)
