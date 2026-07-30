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

logger = logging.getLogger("tilevision.presentation.workers.search_worker")

# Heartbeat interval while Search is running (stall detector uses this).
_HEARTBEAT_INTERVAL_S = 5.0


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

    def run(self) -> None:
        """Execute the search in the background thread."""
        logger.info("Search QThread started for query image: %s", self._query_image_path)
        start_time = time.monotonic()
        stop_heartbeat = threading.Event()

        def _heartbeat_loop() -> None:
            while not stop_heartbeat.wait(_HEARTBEAT_INTERVAL_S):
                if self.isInterruptionRequested():
                    return
                self.search_heartbeat.emit()

        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            name="search-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()

        try:
            if self.isInterruptionRequested():
                logger.info("Search QThread interrupted before execute.")
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
                return

            self.search_progress.emit("Running AI match (this may take a moment on CPU)…")
            self.search_heartbeat.emit()

            results = None
            last_error: Exception | None = None
            for attempt in (1, 2, 3):
                if self.isInterruptionRequested():
                    logger.info("Search QThread interrupted before attempt %s.", attempt)
                    return
                try:
                    results = self._use_case.execute(
                        self._query_image_path,
                        top_k=self._top_k,
                        filters=self._filters,
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
                logger.info(
                    "Search QThread interrupted after %.3fs — suppressing result emit.",
                    elapsed,
                )
                return

            self.search_progress.emit("Building results…")
            self.search_heartbeat.emit()
            logger.info(
                "Search QThread finished in %.3fs. Results: %d",
                elapsed,
                len(results),
            )
            self.search_timed.emit(elapsed)
            self.search_completed.emit(results)
        except Exception as e:
            if self.isInterruptionRequested():
                logger.info(
                    "Search QThread interrupted during failure for '%s': %s",
                    self._query_image_path,
                    e,
                )
                return
            logger.error("Search worker failed for query '%s': %s", self._query_image_path, e)
            self.search_failed.emit(str(e))
        finally:
            stop_heartbeat.set()
