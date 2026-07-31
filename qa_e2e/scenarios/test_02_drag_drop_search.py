"""Customer drags a tile photo onto Search and gets real matches."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from qa_e2e.framework.expectations import evaluate_from_manifest
from qa_e2e.framework.failures import detect_after_search, detect_ui_freeze
from qa_e2e.framework.readiness import wait_for_search_pipeline
from src.utils import search_stages

pytestmark = pytest.mark.qa_e2e


def test_drag_drop_search_full_pipeline(session, driver, catalog_indexed, record_expectation):
    queries = [q for q in session.expected_manifest["queries"] if q["kind"] == "crop_match"]
    assert queries, "fixture must provide crop_match queries"
    q = queries[0]

    action = session.artifacts.begin("Drag & Drop search", detail=q["id"])
    since = time.time()
    driver.drag_drop_image(Path(q["path"]))

    missing = wait_for_search_pipeline(session, since=since, timeout=300.0)
    freeze = detect_ui_freeze(session)
    scan = detect_after_search(session, since=since, expect_results=True)

    paths = driver.visible_result_paths()
    # Prefer viewmodel results if table path column empty briefly
    if not paths and session.search_viewmodel.last_results:
        paths = [r.tile.file_path for r in session.search_viewmodel.last_results]

    exp = record_expectation(
        evaluate_from_manifest(session.expected_manifest, q["id"], paths)
    )

    # Required production stages
    required = [
        search_stages.STAGE_FAISS_SEARCH,
        search_stages.STAGE_SQLITE_HYDRATE,
        search_stages.STAGE_RERANK_COMPLETE,
        search_stages.STAGE_RESULTS_READY,
    ]
    stage_ok = all(session.logs.contains(s, since=since) for s in required)
    # Embedding: generated or cache hit
    embed_ok = session.logs.contains(search_stages.STAGE_EMBEDDING_GENERATED, since=since) or session.logs.contains(
        search_stages.STAGE_EMBEDDING_CACHE_HIT, since=since
    )
    norm_ok = session.logs.contains(search_stages.STAGE_EMBEDDING_NORMALIZED, since=since) or session.logs.contains(
        search_stages.STAGE_EMBEDDING_CACHE_HIT, since=since
    )

    ok = (
        scan.ok
        and freeze.ok
        and stage_ok
        and embed_ok
        and norm_ok
        and exp.ok
        and not missing
    )
    session.artifacts.end(
        action,
        ok=ok,
        detail=exp.detail,
        screenshot_widget=session.main_window,
        metrics={
            "missing_stages": missing,
            "findings": [f.message for f in scan.findings],
            "status": driver.search_status(),
            "result_count": driver.result_count_ui(),
            "elapsed_s": getattr(session.search_viewmodel, "_last_elapsed_seconds", 0.0),
            "top_product": exp.top_product,
        },
    )
    assert freeze.ok, freeze.findings
    assert scan.ok, [f.message for f in scan.findings]
    assert stage_ok, f"missing required stages; log snip={session.logs.messages(since=since)[-30:]}"
    assert embed_ok and norm_ok
    assert exp.ok, exp.detail


def test_multiple_searches_in_a_row(session, driver, catalog_indexed, record_expectation):
    queries = [q for q in session.expected_manifest["queries"] if q["kind"] == "crop_match"][:3]
    for q in queries:
        action = session.artifacts.begin("Multiple search", detail=q["id"])
        since = time.time()
        driver.drag_drop_image(Path(q["path"]))
        driver.wait_search_settled(timeout=300.0)
        paths = driver.visible_result_paths() or [
            r.tile.file_path for r in session.search_viewmodel.last_results
        ]
        exp = record_expectation(
            evaluate_from_manifest(session.expected_manifest, q["id"], paths)
        )
        session.human.think(0.3, 0.9)
        session.artifacts.end(
            action,
            ok=exp.ok,
            detail=exp.detail,
            screenshot_widget=session.main_window,
            metrics={"elapsed_s": getattr(session.search_viewmodel, "_last_elapsed_seconds", 0.0)},
        )
        assert exp.ok, exp.detail
