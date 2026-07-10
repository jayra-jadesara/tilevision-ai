"""
Search worker module for TileVision AI.

Implements a PySide6 QThread class to execute visual similarity search
(embedding extraction + FAISS query + metadata hydration) in the background,
keeping the UI thread fully responsive while the CLIP model runs inference.
"""

import logging
import time
from typing import Optional

from PySide6.QtCore import QThread, Signal

from src.core.use_cases.search_tiles import SearchTilesUseCase

logger = logging.getLogger("tilevision.presentation.workers.search_worker")


class SearchWorker(QThread):
    """
    Background worker thread that executes a single visual similarity search.

    Each search creates a fresh, single-shot worker instance (searches are
    not pausable/cancellable/long-running like folder indexing), so there is
    no shared mutable worker state to race on between instances.
    """

    # Signal payload: (results) — a List[SearchResult], passed as a generic
    # Python object since SearchResult is a plain dataclass, not a QObject.
    search_completed = Signal(list)

    # Signal payload: (error_message)
    search_failed = Signal(str)

    # Signal payload: (elapsed_seconds) — emitted alongside search_completed
    # so the UI/logs can track against the <2s performance target.
    search_timed = Signal(float)

    def __init__(
        self,
        use_case: SearchTilesUseCase,
        query_image_path: str,
        top_k: int,
        filters: Optional[dict] = None,
    ) -> None:
        """
        Initialize the search background worker.

        Args:
            use_case: Fully configured SearchTilesUseCase.
            query_image_path: Absolute path to the query image file.
            top_k: Maximum number of results to return.
            filters: Optional dict of metadata field -> required value
                (Feature 8), e.g. {"brand": "Kajaria"}.
        """
        super().__init__()
        self._use_case = use_case
        self._query_image_path = query_image_path
        self._top_k = top_k
        self._filters = filters or {}

    def run(self) -> None:
        """Execute the search in the background thread."""
        logger.info(f"Search QThread started for query image: {self._query_image_path}")
        start_time = time.monotonic()

        try:
            results = self._use_case.execute(
                self._query_image_path, top_k=self._top_k, filters=self._filters
            )
            elapsed = time.monotonic() - start_time

            logger.info(f"Search QThread finished in {elapsed:.3f}s. Results: {len(results)}")
            if elapsed > 2.0:
                logger.warning(
                    f"Search exceeded the 2-second performance target: {elapsed:.3f}s "
                    f"for query '{self._query_image_path}'."
                )

            self.search_timed.emit(elapsed)
            self.search_completed.emit(results)
        except Exception as e:
            logger.error(f"Search worker failed for query '{self._query_image_path}': {e}")
            self.search_failed.emit(str(e))
