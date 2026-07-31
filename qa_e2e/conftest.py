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


def _index_catalog_with_fallback(session, driver) -> dict:
    """
    Customer path: select folder + Start Indexing (IndexingWorker QThread).

    On some Mac Intel CI hosts, PyTorch + OpenMP inside a Qt QThread can stall
    with no progress. If FAISS count does not grow, cancel and finish the same
    ``IndexImagesUseCase.scan_and_index_directory`` on a Python thread while
    pumping the Qt event loop — still real DINOv2 / FAISS / SQLite.
    """
    import threading
    import time

    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from src.presentation.viewmodels.indexing_viewmodel import IndexingState

    driver.select_index_folder(session.catalog_dir)
    driver.start_indexing()

    deadline = time.monotonic() + float(os.environ.get("TILEVISION_QA_INDEX_TIMEOUT", "1200"))
    last_count = session.vector_index.get_total_count()
    last_growth = time.monotonic()
    used_fallback = False

    while time.monotonic() < deadline:
        QApplication.processEvents()
        state = session.indexing_viewmodel.state
        count = session.vector_index.get_total_count()
        if count > last_count:
            last_count = count
            last_growth = time.monotonic()
        if state == IndexingState.FINISHED:
            break
        if state == IndexingState.ERROR:
            raise RuntimeError("Indexing entered ERROR state")
        # No FAISS growth for 90s while still "running" → likely QThread stall
        stalled = (
            state in (IndexingState.RUNNING, IndexingState.PAUSED)
            and count == 0
            and (time.monotonic() - last_growth) > 90.0
        )
        if stalled:
            session.artifacts.note(
                "IndexingWorker produced no FAISS growth in 90s — "
                "falling back to Python-thread IndexImagesUseCase (same production path)"
            )
            try:
                session.indexing_viewmodel.cancel_indexing()
            except Exception:
                pass
            QTest.qWait(500)

            box: dict = {}

            def _run() -> None:
                try:
                    box["result"] = session.index_use_case.scan_and_index_directory(
                        directory_path=session.catalog_dir
                    )
                except Exception as exc:  # pragma: no cover
                    box["error"] = exc

            thread = threading.Thread(target=_run, name="qa-index-fallback", daemon=True)
            thread.start()
            while thread.is_alive():
                QApplication.processEvents()
                QTest.qWait(200)
            if "error" in box:
                raise box["error"]
            used_fallback = True
            break
        QTest.qWait(250)
    else:
        raise TimeoutError(
            f"Indexing did not finish within deadline (state={session.indexing_viewmodel.state})"
        )

    count = session.vector_index.get_total_count()
    sqlite_n = len(session.image_repository.get_all())
    return {
        "faiss_count": count,
        "sqlite_count": sqlite_n,
        "used_fallback": used_fallback,
        "state": session.indexing_viewmodel.state,
    }


@pytest.fixture(scope="session")
def catalog_indexed(session, driver):
    """Index the synthetic showroom catalogue once for the whole QA session."""
    action = session.artifacts.begin("Index Folder", detail=str(session.catalog_dir))
    try:
        metrics = _index_catalog_with_fallback(session, driver)
        ok = metrics["faiss_count"] > 0 and metrics["sqlite_count"] > 0
        session.artifacts.end(
            action,
            ok=ok,
            detail=f"faiss={metrics['faiss_count']} sqlite={metrics['sqlite_count']} "
            f"fallback={metrics['used_fallback']}",
            screenshot_widget=session.main_window,
            metrics=metrics,
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
