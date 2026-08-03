"""Complete a search via the production use case when SearchWorker stalls."""

from __future__ import annotations

import logging
import threading
import time
from typing import List

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src.core.models import SearchResult
from src.presentation.viewmodels.search_viewmodel import SearchState

logger = logging.getLogger("tilevision.qa_e2e.search_fallback")


def complete_search_via_use_case(session, image_path: str, *, timeout: float = 600.0) -> List[SearchResult]:
    """
    Run ``SearchTilesUseCase.execute`` on a Python thread and push results
    into the live SearchViewModel so the UI table updates.

    Used when SearchWorker (QThread) stalls on Mac Intel CI — same DINOv2 /
    FAISS / SQLite / hybrid rerank path as production.
    """
    box: dict = {}

    def _run() -> None:
        try:
            box["results"] = session.search_use_case.execute(
                query_image_path=image_path,
                top_k=int(session.settings.top_k),
            )
        except Exception as exc:  # pragma: no cover
            box["error"] = exc

    vm = session.search_viewmodel
    worker = getattr(vm, "_worker", None)
    if worker is not None:
        try:
            if worker.isRunning():
                worker.requestInterruption()
                try:
                    worker.wait(15_000)
                except Exception:
                    pass
        except Exception:
            pass
    vm._worker = None

    thread = threading.Thread(target=_run, name="qa-search", daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while thread.is_alive():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Search use-case fallback timed out for {image_path}")
        QApplication.processEvents()
        QTest.qWait(200)

    if "error" in box:
        raise box["error"]

    results: List[SearchResult] = list(box.get("results") or [])
    gen = int(getattr(vm, "_search_generation", 0))
    vm._last_query_path = image_path
    # We are already on the Qt GUI thread inside the test driver.
    vm._on_search_completed(results, gen)
    QApplication.processEvents()
    QTest.qWait(200)
    if vm.state == SearchState.SEARCHING:
        vm._set_state(SearchState.RESULTS if results else SearchState.NO_RESULTS)
        vm.results_ready.emit(results)
        QApplication.processEvents()
    logger.info("[QA] Search fallback delivered %d result(s) for %s", len(results), image_path)
    return results
