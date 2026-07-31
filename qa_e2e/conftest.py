"""
Pytest fixtures for TileVision human-like E2E QA.

Session-scoped real app (DINOv2 + FAISS + SQLite + MainWindow).
Normal ``tests/`` CI does not import this package.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Must set Qt platform before QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("TILEVISION_DEV_MODE", "1")
os.environ.setdefault("TILEVISION_LOG_LEVEL", "INFO")
os.environ.setdefault("TILEVISION_PROFILE", "1")

from qa_e2e.framework.expectations import ExpectationResult  # noqa: E402
from qa_e2e.framework.harness import launch_customer_app  # noqa: E402
from qa_e2e.framework.readiness import probe_readiness  # noqa: E402
from qa_e2e.framework.report import write_html_report  # noqa: E402
from qa_e2e.framework.ui_driver import UIDriver  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "qa_e2e: human-like end-to-end QA against the real app stack"
    )


@pytest.fixture(scope="session")
def qa_work_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("tilevision_qa_home")


@pytest.fixture(scope="session")
def qa_artifact_dir() -> Path:
    env = os.environ.get("TILEVISION_QA_OUT")
    if env:
        out = Path(env)
    else:
        out = ROOT / "qa_e2e" / "artifacts" / "latest"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture(scope="session")
def session(qa_work_dir, qa_artifact_dir):
    """Launch once: real model warm-up is expensive."""
    try:
        app_session = launch_customer_app(
            work_dir=qa_work_dir,
            artifact_dir=qa_artifact_dir,
            human_seed=int(os.environ.get("TILEVISION_QA_SEED", "42")),
            human_speed=float(os.environ.get("TILEVISION_QA_SPEED", "1.8")),
            catalog_tiles=int(os.environ.get("TILEVISION_QA_TILES", "12")),
        )
    except Exception as exc:
        pytest.skip(f"Unable to launch real customer app for E2E QA: {exc}")
    yield app_session
    # Write report at end of session
    readiness = probe_readiness(app_session, require_catalog=False)
    expectations = getattr(app_session, "_qa_expectations", [])
    log_lines = app_session.logs.messages()[-400:]
    final_pass = (
        not app_session.artifacts.failures
        and all(e.get("ok", False) for e in expectations)
        if expectations
        else not app_session.artifacts.failures
    )
    # If any pytest failures happened, still emit report; verdict from artifacts.
    write_html_report(
        out_dir=qa_artifact_dir,
        collector=app_session.artifacts,
        readiness=asdict(readiness),
        expectations=expectations,
        log_excerpt=log_lines,
        final_pass=final_pass and readiness.ok,
    )
    app_session.close()


@pytest.fixture(scope="session")
def driver(session) -> UIDriver:
    return UIDriver(session)


@pytest.fixture(scope="session")
def catalog_indexed(session, driver):
    """Index the synthetic showroom catalogue once for the whole QA session."""
    action = session.artifacts.begin("Index Folder", detail=str(session.catalog_dir))
    try:
        driver.select_index_folder(session.catalog_dir)
        driver.start_indexing()
        driver.wait_indexing_done(timeout=900.0)
        count = session.vector_index.get_total_count()
        sqlite_n = len(session.image_repository.get_all())
        ok = count > 0 and sqlite_n > 0
        session.artifacts.end(
            action,
            ok=ok,
            detail=f"faiss={count} sqlite={sqlite_n}",
            screenshot_widget=session.main_window,
            metrics={"faiss_count": count, "sqlite_count": sqlite_n},
        )
        if not ok:
            pytest.fail("Catalog indexing produced empty FAISS/SQLite")
    except Exception as exc:
        session.artifacts.end(action, ok=False, detail=str(exc), screenshot_widget=session.main_window)
        raise
    return True


@pytest.fixture
def record_expectation(session):
    def _record(result: ExpectationResult) -> ExpectationResult:
        bucket = getattr(session, "_qa_expectations", None)
        if bucket is None:
            bucket = []
            session._qa_expectations = bucket
        bucket.append(
            {
                "query_id": result.query_id,
                "expected_product": result.expected_product,
                "top_product": result.top_product,
                "rank_of_expected": result.rank_of_expected,
                "ok": result.ok,
                "detail": result.detail,
            }
        )
        return result

    return _record
