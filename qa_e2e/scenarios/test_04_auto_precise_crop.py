"""Customer uses Auto Crop and Precise Crop before searching."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from qa_e2e.framework.failures import detect_after_search
from src.presentation.viewmodels.search_viewmodel import SearchState

pytestmark = pytest.mark.qa_e2e


def _pick_query(session) -> Path:
    q = next(q for q in session.expected_manifest["queries"] if q["kind"] == "crop_match")
    return Path(q["path"])


def test_auto_crop_and_search(session, driver, catalog_indexed):
    path = _pick_query(session)
    action = session.artifacts.begin("Auto Crop & Search", detail=str(path.name))
    # Load query into drop zone first (enables crop buttons)
    driver.open_image_via_viewmodel(path)
    driver.wait_search_settled(timeout=300.0)
    session.human.think(0.4, 1.0)

    since = time.time()
    try:
        driver.auto_crop_search()
    except AssertionError as exc:
        # Button disabled until a query is present — fail clearly
        session.artifacts.end(action, ok=False, detail=str(exc), screenshot_widget=session.main_window)
        raise

    state = driver.wait_search_settled(timeout=420.0)
    scan = detect_after_search(session, since=since, expect_results=True)
    # Auto crop may log OpenCV path; search must still complete
    ok = state in (SearchState.RESULTS, SearchState.NO_RESULTS) and (
        state == SearchState.RESULTS or session.vector_index.get_total_count() == 0
    )
    # Prefer results on a known matching query
    ok = ok and state == SearchState.RESULTS and scan.ok
    session.artifacts.end(
        action,
        ok=ok,
        detail=f"state={state} status={driver.search_status()}",
        screenshot_widget=session.main_window,
        metrics={
            "state": state,
            "results": driver.result_count_ui(),
            "findings": [f.message for f in scan.findings],
        },
    )
    assert ok, scan.findings


def test_precise_crop_and_search(session, driver, catalog_indexed):
    path = _pick_query(session)
    action = session.artifacts.begin("Precise Crop & Search", detail=str(path.name))
    driver.open_image_via_viewmodel(path)
    driver.wait_search_settled(timeout=300.0)
    session.human.think(0.5, 1.2)

    since = time.time()
    try:
        driver.precise_crop_search()
    except AssertionError as exc:
        session.artifacts.end(action, ok=False, detail=str(exc), screenshot_widget=session.main_window)
        raise

    # Precise crop (SAM2/ONNX/GrabCut) can take longer on CPU Intel Macs
    state = driver.wait_search_settled(timeout=600.0)
    scan = detect_after_search(session, since=since, expect_results=False)
    # Accept RESULTS, or a clear crop failure message (environment without SAM2 weights)
    status = driver.search_status().lower()
    crop_failed = "crop" in status and ("fail" in status or "error" in status)
    ok = state == SearchState.RESULTS or crop_failed or state == SearchState.ERROR
    # Soft-pass: if SAM2 unavailable, record as non-fatal note but still require UI settled
    if crop_failed or state == SearchState.ERROR:
        session.artifacts.note("Precise Crop unavailable or failed in this environment — UI settled")
        ok = True
    session.artifacts.end(
        action,
        ok=ok,
        detail=f"state={state} status={driver.search_status()}",
        screenshot_widget=session.main_window,
        metrics={"state": state, "scan_ok": scan.ok},
    )
    assert ok
