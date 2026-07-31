"""Longer human-like session: navigate, zoom/scroll pauses, re-search, change crop."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from qa_e2e.framework.expectations import evaluate_from_manifest
from qa_e2e.framework.failures import detect_ui_freeze

pytestmark = pytest.mark.qa_e2e


def test_realistic_showroom_session(session, driver, catalog_indexed, record_expectation):
    action = session.artifacts.begin("Human showroom session")
    findings = []

    # Open app feeling: wander nav
    driver.goto_dashboard()
    session.human.think(0.5, 1.2)
    driver.goto_settings()
    session.human.wander(session.main_window, steps=6)
    session.human.think(0.4, 0.9)
    driver.goto_index()
    session.human.think(0.3, 0.7)
    driver.goto_search()

    queries = [q for q in session.expected_manifest["queries"] if q["kind"] == "crop_match"]
    # First search
    q1 = queries[0]
    driver.drag_drop_image(Path(q1["path"]))
    driver.wait_search_settled(timeout=300.0)
    session.human.scroll(driver.results_table())
    session.human.think(0.6, 1.4)

    # Search again with another photo
    q2 = queries[1]
    driver.drag_drop_image(Path(q2["path"]))
    driver.wait_search_settled(timeout=300.0)
    paths = driver.visible_result_paths() or [
        r.tile.file_path for r in session.search_viewmodel.last_results
    ]
    exp = record_expectation(
        evaluate_from_manifest(session.expected_manifest, q2["id"], paths)
    )
    if not exp.ok:
        findings.append(exp.detail)

    # Change crop mode and search again
    session.human.think(0.4, 1.0)
    try:
        driver.auto_crop_search()
        driver.wait_search_settled(timeout=420.0)
    except Exception as exc:
        findings.append(f"auto crop soft-fail: {exc}")

    # Random click timings / navigate away and back
    driver.goto_dashboard()
    session.human.think(0.3, 0.8)
    driver.goto_search()
    session.human.wander(driver.drop_zone(), steps=4)

    freeze = detect_ui_freeze(session)
    if not freeze.ok:
        findings.extend(f.message for f in freeze.findings)

    ok = exp.ok and freeze.ok
    session.artifacts.end(
        action,
        ok=ok,
        detail="; ".join(findings) if findings else "session completed",
        screenshot_widget=session.main_window,
        metrics={"findings": findings, "t": time.time()},
    )
    assert ok, findings
