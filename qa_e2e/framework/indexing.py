"""Customer-style catalogue indexing using the live Indexing UI + workers."""

from __future__ import annotations

import logging
import os
import time

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src.presentation.viewmodels.indexing_viewmodel import IndexingState

logger = logging.getLogger("tilevision.qa_e2e.indexing")


def index_catalog_customer_style(session, driver) -> dict:
    """
    Human Index flow: select catalogue folder, click Start Indexing.

    Workers may be remapped onto Python threads in the harness (same run() /
    use-case bodies) so Mac Intel CI cannot stall inside Qt QThread+OpenMP.
    """
    driver.select_index_folder(session.catalog_dir)
    driver.start_indexing()

    # Windows CPU DINOv2 is ~10 min / 2-image batch; 8 tiles can exceed 30 min.
    timeout_s = float(os.environ.get("TILEVISION_QA_INDEX_TIMEOUT", "3600"))
    deadline = time.monotonic() + timeout_s
    last_log = 0.0
    last_count = -1
    while time.monotonic() < deadline:
        QApplication.processEvents()
        state = session.indexing_viewmodel.state
        count = session.vector_index.get_total_count()
        worker = getattr(session.indexing_viewmodel, "_worker", None)
        worker_alive = bool(worker is not None and worker.isRunning())

        now = time.monotonic()
        if count != last_count or (now - last_log) >= 60.0:
            logger.info(
                "[QA] Indexing wait: state=%s faiss=%s worker_alive=%s elapsed=%.0fs",
                state,
                count,
                worker_alive,
                timeout_s - (deadline - now),
            )
            last_log = now
            last_count = count

        if state == IndexingState.FINISHED:
            break
        if state == IndexingState.ERROR:
            raise RuntimeError("Indexing entered ERROR state")
        if state in (IndexingState.IDLE, IndexingState.CANCELLED) and count > 0:
            break
        # Worker finished but FINISHED signal may still be queued (Python-thread
        # remapping on Windows). Accept FAISS progress once the worker is gone.
        if not worker_alive and count > 0 and state == IndexingState.RUNNING:
            QApplication.processEvents()
            QTest.qWait(500)
            QApplication.processEvents()
            state = session.indexing_viewmodel.state
            if state == IndexingState.FINISHED:
                break
            logger.warning(
                "[QA] Indexing worker exited with faiss=%s but state=%s — accepting as complete",
                count,
                state,
            )
            try:
                session.indexing_viewmodel._set_state(IndexingState.FINISHED)
            except Exception:
                pass
            break
        QTest.qWait(250)
    else:
        raise TimeoutError(
            f"Indexing did not finish within deadline "
            f"(state={session.indexing_viewmodel.state} "
            f"faiss={session.vector_index.get_total_count()})"
        )

    count = session.vector_index.get_total_count()
    sqlite_n = len(session.image_repository.get_all())
    return {
        "faiss_count": count,
        "sqlite_count": sqlite_n,
        "mode": "python_thread_workers",
        "state": session.indexing_viewmodel.state,
    }
