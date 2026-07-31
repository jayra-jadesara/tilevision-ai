"""Customer opens an image via Browse (file dialog path) and searches."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from qa_e2e.framework.expectations import evaluate_from_manifest
from qa_e2e.framework.failures import detect_after_search

pytestmark = pytest.mark.qa_e2e


def test_open_image_dialog_path_searches(session, driver, catalog_indexed, record_expectation):
    """
    Headless runners cannot drive the native QFileDialog, so we exercise the
    same DropZone → _on_image_chosen path the Browse click uses after the
    dialog returns a path (identical production call chain).
    """
    q = next(q for q in session.expected_manifest["queries"] if q["id"] == "q_jpg")
    action = session.artifacts.begin("Open Image (browse path)", detail=q["id"])
    since = time.time()
    driver.open_image_via_viewmodel(Path(q["path"]))
    state = driver.wait_search_settled(timeout=300.0)
    scan = detect_after_search(session, since=since, expect_results=True)
    paths = driver.visible_result_paths() or [
        getattr(r, "image_path", getattr(r, "file_path", ""))
        for r in getattr(session.search_viewmodel, "_last_results", [])
    ]
    exp = record_expectation(
        evaluate_from_manifest(session.expected_manifest, q["id"], paths)
    )
    ok = scan.ok and exp.ok and state != "error"
    session.artifacts.end(
        action,
        ok=ok,
        detail=exp.detail,
        screenshot_widget=session.main_window,
        metrics={"state": state, "status": driver.search_status()},
    )
    assert ok, (scan.findings, exp.detail)
