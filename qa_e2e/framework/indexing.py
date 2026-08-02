"""Customer-style catalogue indexing using the live Indexing UI + workers."""

from __future__ import annotations

import os
import time

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src.presentation.viewmodels.indexing_viewmodel import IndexingState


def index_catalog_customer_style(session, driver) -> dict:
    """
    Human Index flow: select catalogue folder, click Start Indexing.

    Workers may be remapped onto Python threads in the harness (same run() /
    use-case bodies) so Mac Intel CI cannot stall inside Qt QThread+OpenMP.
    """
    driver.select_index_folder(session.catalog_dir)
    driver.start_indexing()

    deadline = time.monotonic() + float(os.environ.get("TILEVISION_QA_INDEX_TIMEOUT", "1800"))
    while time.monotonic() < deadline:
        QApplication.processEvents()
        state = session.indexing_viewmodel.state
        if state == IndexingState.FINISHED:
            break
        if state == IndexingState.ERROR:
            raise RuntimeError("Indexing entered ERROR state")
        if state in (IndexingState.IDLE, IndexingState.CANCELLED) and session.vector_index.get_total_count() > 0:
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
        "mode": "python_thread_workers",
        "state": session.indexing_viewmodel.state,
    }
