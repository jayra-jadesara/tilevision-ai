"""
Search worker module for TileVision AI.

Implements a PySide6 QThread class to execute visual similarity search
(embedding extraction + FAISS query + metadata hydration) in the background,
keeping the UI thread fully responsive while DINOv2 runs inference.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from PySide6.QtCore import QThread, Signal

from src.core.use_cases.search_tiles import SearchTilesUseCase
from src.utils.search_stages import log_search_stage

logger = logging.getLogger("tilevision.presentation.workers.search_worker")

# Heartbeat interval while Search is running (stall detector uses this).
_HEARTBEAT_INTERVAL_S = 5.0
_CANCELLED_MESSAGE = "Search cancelled"


class SearchWorker(QThread):
    """
    Background worker thread that executes a single visual similarity search.

    Each search creates a fresh, single-shot worker instance.
    """

    search_completed = Signal(list)
    search_failed = Signal(str)
    search_timed = Signal(float)
    # Human-readable stage for the status line (never abort on its own).
    search_progress = Signal(str)
    # Periodic liveness pulse during long DINOv2 / FAISS work.
    search_heartbeat = Signal()

    def __init__(
        self,
        use_case: SearchTilesUseCase,
        query_image_path: str,
        top_k: int,
        filters: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self._use_case = use_case
        self._query_image_path = query_image_path
        self._top_k = top_k
        self._filters = filters or {}

    def _emit_cancelled(self, where: str) -> None:
        """Always notify the UI when the worker exits due to interruption."""
        logger.warning(
            "Search QThread cancelled at %s for %s — emitting failure so UI cannot hang.",
            where,
            self._query_image_path,
        )
        self.search_failed.emit(_CANCELLED_MESSAGE)

    def run(self) -> None:
        """Execute the search in the background thread."""
        logger.info("Search worker started for query image: %s", self._query_image_path)
        start_time = time.monotonic()
        stop_heartbeat = threading.Event()
        # Capture the main-thread receiver affinity via Queued signal emits.
        # Heartbeat runs on a helper thread; Qt queues delivery to the ViewModel.
        worker_ref = self

        def _heartbeat_loop() -> None:
            while not stop_heartbeat.wait(_HEARTBEAT_INTERVAL_S):
                if worker_ref.isInterruptionRequested():
                    return
                try:
                    worker_ref.search_heartbeat.emit()
                except RuntimeError:
                    # Worker may already be deleted — stop quietly.
                    return

        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            name="search-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()

        try:
            if self.isInterruptionRequested():
                self._emit_cancelled("before_execute")
                return

            from src.ai.inference_guard import (
                InferenceBusyError,
                wait_until_inference_idle,
            )

            self.search_progress.emit("Pausing Indexing so Search can run…")
            self.search_heartbeat.emit()
            # Let any in-flight indexing forward finish / yield.
            idle = wait_until_inference_idle(max_wait_s=90.0)
            if not idle:
                logger.warning(
                    "Inference still busy after wait — Search will retry on lock."
                )
            if self.isInterruptionRequested():
                self._emit_cancelled("after_idle_wait")
                return

            self.search_progress.emit("Running AI match (this may take a moment on CPU)…")
            self.search_heartbeat.emit()

            def _on_stage(message: str) -> None:
                if self.isInterruptionRequested():
                    return
                self.search_progress.emit(message)
                self.search_heartbeat.emit()

            results = None
            last_error: Exception | None = None
            for attempt in (1, 2, 3):
                if self.isInterruptionRequested():
                    self._emit_cancelled(f"before_attempt_{attempt}")
                    return
                try:
                    results = self._use_case.execute(
                        self._query_image_path,
                        top_k=self._top_k,
                        filters=self._filters,
                        on_stage=_on_stage,
                    )
                    last_error = None
                    break
                except InferenceBusyError as exc:
                    last_error = exc
                    logger.warning(
                        "Search attempt %s blocked by busy AI engine — waiting…",
                        attempt,
                    )
                    self.search_progress.emit(
                        f"AI engine busy (attempt {attempt}/3) — waiting for Indexing…"
                    )
                    self.search_heartbeat.emit()
                    wait_until_inference_idle(max_wait_s=45.0)
                    time.sleep(0.5)
                    continue

            if last_error is not None:
                raise last_error
            assert results is not None

            elapsed = time.monotonic() - start_time
            if self.isInterruptionRequested():
                self._emit_cancelled("after_execute")
                return

            self.search_progress.emit("Building results…")
            self.search_heartbeat.emit()
            log_search_stage(
                logger,
                "Worker emitting results",
                detail=f"{len(results)} in {elapsed:.3f}s",
            )
            logger.info(
                "Search worker finished in %.3fs. Results: %d",
                elapsed,
                len(results),
            )
            self.search_timed.emit(elapsed)
            self.search_completed.emit(results)
        except Exception as e:
            if self.isInterruptionRequested():
                self._emit_cancelled("during_failure")
                return
            logger.error("Search worker failed for query '%s': %s", self._query_image_path, e)
            self.search_failed.emit(str(e))
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=1.0)
