"""
Search ViewModel module for TileVision AI.

Manages the state of a visual similarity search: accepting a query image
(drag-and-drop or browse), running it on a background worker thread, and
exposing results/status to the SearchView through Qt signals.
"""

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, Signal, Slot

from src.core.models import SearchResult
from src.core.use_cases.search_tiles import SearchTilesUseCase
from src.presentation.workers.search_worker import SearchWorker

logger = logging.getLogger("tilevision.presentation.viewmodels.search_viewmodel")


class SearchState:
    """Enumeration of valid search lifecycle states."""
    IDLE = "idle"
    SEARCHING = "searching"
    RESULTS = "results"
    NO_RESULTS = "no_results"
    ERROR = "error"


class SearchViewModel(QObject):
    """
    ViewModel coordinating visual similarity search for the SearchView.

    Owns a SearchTilesUseCase and drives a background SearchWorker per
    query. A new search is rejected while one is already in flight (rather
    than allowing concurrent searches), which also sidesteps the kind of
    stale-worker-reference race that folder indexing's pause/cancel flow
    has to guard against — there's nothing to race when only one worker
    can ever be active at a time.
    """

    state_changed = Signal(str)
    results_ready = Signal(list)  # List[SearchResult]
    status_message = Signal(str)
    search_error = Signal(str)
    query_image_selected = Signal(str)  # absolute path of the chosen query image

    def __init__(self, use_case: SearchTilesUseCase, default_top_k: int = 20) -> None:
        """
        Initialize the SearchViewModel.

        Args:
            use_case: Fully configured SearchTilesUseCase.
            default_top_k: Default maximum number of results to request.
        """
        super().__init__()
        self._use_case = use_case
        self._top_k = default_top_k
        self._state = SearchState.IDLE
        self._worker: Optional[SearchWorker] = None
        self._last_results: List[SearchResult] = []
        self._last_query_path: Optional[str] = None

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_searching(self) -> bool:
        return self._state == SearchState.SEARCHING

    @property
    def last_results(self) -> List[SearchResult]:
        return list(self._last_results)

    @property
    def top_k(self) -> int:
        return self._top_k

    @top_k.setter
    def top_k(self, value: int) -> None:
        self._top_k = max(1, int(value))

    # ── Public Slots (invoked from the View) ────────────────────────────────

    @Slot(str)
    def search_by_image(self, image_path: str) -> None:
        """
        Kick off a background visual similarity search for the given image.

        Ignored (with a log warning) if a search is already in progress —
        the UI should disable the drop zone/browse button while searching
        to prevent this in practice, but the guard is enforced here too.

        Args:
            image_path: Absolute path to the query image (dropped or browsed).
        """
        if self._state == SearchState.SEARCHING:
            logger.warning("Search already in progress; ignoring new search request.")
            return

        path = Path(image_path)
        if not path.exists() or not path.is_file():
            self._set_state(SearchState.ERROR)
            self.search_error.emit(f"Selected file does not exist: {image_path}")
            self.status_message.emit("Search failed: file not found.")
            return

        self._last_query_path = str(path)
        self.query_image_selected.emit(str(path))

        self._set_state(SearchState.SEARCHING)
        self.status_message.emit(f"Searching for tiles similar to '{path.name}'...")

        self._worker = SearchWorker(self._use_case, str(path), self._top_k)
        self._worker.search_completed.connect(self._on_search_completed)
        self._worker.search_failed.connect(self._on_search_failed)
        self._worker.search_timed.connect(self._on_search_timed)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    @Slot()
    def clear_results(self) -> None:
        """Reset search state back to idle and clear any displayed results."""
        self._last_results = []
        self._last_query_path = None
        self._set_state(SearchState.IDLE)
        self.results_ready.emit([])
        self.status_message.emit("Ready. Drag an image or click Browse to search.")

    # ── Internal State Management ────────────────────────────────────────

    def _set_state(self, new_state: str) -> None:
        if self._state != new_state:
            self._state = new_state
            self.state_changed.emit(new_state)

    # ── Worker Signal Handlers ───────────────────────────────────────────

    @Slot(list)
    def _on_search_completed(self, results: List[SearchResult]) -> None:
        self._last_results = results
        self._worker = None

        if results:
            self._set_state(SearchState.RESULTS)
            self.status_message.emit(f"Found {len(results)} similar tile(s).")
        else:
            self._set_state(SearchState.NO_RESULTS)
            self.status_message.emit("No similar tiles found in the indexed catalog.")

        self.results_ready.emit(results)

    @Slot(str)
    def _on_search_failed(self, message: str) -> None:
        self._worker = None
        self._set_state(SearchState.ERROR)
        self.status_message.emit(f"Search failed: {message}")
        self.search_error.emit(message)

    @Slot(float)
    def _on_search_timed(self, elapsed_seconds: float) -> None:
        logger.info(f"Search for '{self._last_query_path}' completed in {elapsed_seconds:.3f}s.")
