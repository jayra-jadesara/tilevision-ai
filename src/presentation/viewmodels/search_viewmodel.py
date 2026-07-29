"""
Search ViewModel module for TileVision AI.

Manages the state of a visual similarity search: accepting a query image
(drag-and-drop or browse), running it on a background worker thread, and
exposing results/status to the SearchView through Qt signals.
"""

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from src.core.models import SearchResult, SearchHistoryEntry
from src.core.use_cases.search_tiles import SearchTilesUseCase
from src.data.repository_interface import ISearchHistoryRepository, IActivityLogRepository
from src.presentation.workers.search_worker import SearchWorker

logger = logging.getLogger("tilevision.presentation.viewmodels.search_viewmodel")

# Hard stop so the UI never sits on "Searching..." forever (DINOv2/MPS/lock hangs).
_SEARCH_TIMEOUT_MS = 90_000


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
    query. A new search is rejected while one is already in flight.
    """

    state_changed = Signal(str)
    results_ready = Signal(list)  # List[SearchResult]
    status_message = Signal(str)
    search_error = Signal(str)
    query_image_selected = Signal(str)  # absolute path of the chosen query image

    filters_available = Signal(dict)  # Dict[str, List[str]] — for populating dropdowns

    # (result_count, elapsed_seconds)
    search_stats_ready = Signal(int, float)

    # List[SearchHistoryEntry]
    search_history_updated = Signal(list)

    def __init__(
        self,
        use_case: SearchTilesUseCase,
        default_top_k: int = 20,
        search_history_repository: Optional[ISearchHistoryRepository] = None,
        activity_log_repository: Optional[IActivityLogRepository] = None,
        search_timeout_ms: int = _SEARCH_TIMEOUT_MS,
    ) -> None:
        super().__init__()
        self._use_case = use_case
        self._top_k = default_top_k
        self._state = SearchState.IDLE
        self._worker: Optional[SearchWorker] = None
        self._last_results: List[SearchResult] = []
        self._last_query_path: Optional[str] = None
        self._active_filters: dict = {}
        self._last_elapsed_seconds: float = 0.0
        self._history_repo = search_history_repository
        self._activity_repo = activity_log_repository
        self._search_generation = 0
        self._search_timeout_ms = max(100, int(search_timeout_ms))
        self._timeout_timer = QTimer(self)
        self._timeout_timer.setSingleShot(True)
        self._timeout_timer.timeout.connect(self._on_search_timeout)

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

    @Slot()
    def load_filter_options(self) -> None:
        try:
            options = self._use_case.get_filter_options()
            self.filters_available.emit(options)
        except Exception as e:
            logger.error(f"Failed to load filter options: {e}")
            self.filters_available.emit({})

    @Slot(str, str)
    def set_filter(self, field: str, value: str) -> None:
        if not value or value.lower() == "any":
            self._active_filters.pop(field, None)
        else:
            self._active_filters[field] = value

        if self._last_query_path and self._state != SearchState.SEARCHING:
            self.search_by_image(self._last_query_path)

    @property
    def active_filters(self) -> dict:
        return dict(self._active_filters)

    @Slot(str)
    def search_by_image(self, image_path: str) -> None:
        if self._state == SearchState.SEARCHING:
            logger.warning("Search already in progress; ignoring new search request.")
            return

        try:
            health = self._use_case.get_index_health()
            indexed = int(getattr(health, "indexed_count", 0) or 0)
            if indexed <= 0:
                self._set_state(SearchState.NO_RESULTS)
                self.results_ready.emit([])
                self.status_message.emit(
                    "No tiles indexed yet. Open Index, add your tile folder, then search again."
                )
                self.search_error.emit(
                    "No tiles are indexed yet.\n\n"
                    "Go to Index, select your catalogue folder, wait until indexing finishes, "
                    "then search again."
                )
                return
            if not health.is_compatible and health.stale_count > 0:
                self._set_state(SearchState.ERROR)
                self.search_error.emit(
                    "Indexed features are outdated. "
                    f"{health.stale_count} of {health.indexed_count} tiles "
                    "need re-indexing.\n\n"
                    "Go to Settings → Rebuild FAISS Index, then search again."
                )
                self.status_message.emit("Search blocked: index is outdated.")
                return
        except Exception as exc:
            logger.warning("Could not verify feature index health: %s", exc)

        path = Path(image_path)
        if not path.exists() or not path.is_file():
            self._set_state(SearchState.ERROR)
            self.search_error.emit(f"Selected file does not exist: {image_path}")
            self.status_message.emit("Search failed: file not found.")
            return

        self._last_query_path = str(path)
        self.query_image_selected.emit(str(path))

        self._search_generation += 1
        generation = self._search_generation

        self._set_state(SearchState.SEARCHING)
        self.status_message.emit(f"Searching for tiles similar to '{path.name}'...")

        self._worker = SearchWorker(self._use_case, str(path), self._top_k, self._active_filters)
        self._worker.search_completed.connect(
            lambda results, gen=generation: self._on_search_completed(results, gen)
        )
        self._worker.search_failed.connect(
            lambda message, gen=generation: self._on_search_failed(message, gen)
        )
        self._worker.search_timed.connect(self._on_search_timed)
        self._worker.finished.connect(self._worker.deleteLater)
        self._timeout_timer.start(self._search_timeout_ms)
        self._worker.start()

    @Slot()
    def clear_results(self) -> None:
        self._timeout_timer.stop()
        self._search_generation += 1
        self._last_results = []
        self._last_query_path = None
        self._set_state(SearchState.IDLE)
        self.results_ready.emit([])
        self.status_message.emit("Ready. Drag an image or click Browse to search.")

    def _set_state(self, new_state: str) -> None:
        if self._state != new_state:
            self._state = new_state
            self.state_changed.emit(new_state)

    def _on_search_completed(self, results: List[SearchResult], generation: int = 0) -> None:
        if generation and generation != self._search_generation:
            logger.info("Ignoring stale search completion (generation %s)", generation)
            return
        self._timeout_timer.stop()
        self._last_results = results
        self._worker = None

        if results:
            self._set_state(SearchState.RESULTS)
            self.status_message.emit(f"Found {len(results)} similar tile(s).")
        else:
            self._set_state(SearchState.NO_RESULTS)
            self.status_message.emit(
                "No similar tiles found in the indexed catalog. "
                "Try another photo, Auto Crop, or check that your catalogue is indexed."
            )

        self.results_ready.emit(results)
        self.search_stats_ready.emit(len(results), self._last_elapsed_seconds)

        if self._history_repo is not None and self._last_query_path:
            try:
                self._history_repo.record_search(
                    self._last_query_path, len(results), self._last_elapsed_seconds
                )
                self.load_search_history()
            except Exception as e:
                logger.error(f"Failed to record search history: {e}")

        if self._activity_repo is not None and self._last_query_path:
            try:
                name = Path(self._last_query_path).name
                self._activity_repo.record_activity(
                    "search", f"Searched with '{name}' — {len(results)} result(s)"
                )
            except Exception as e:
                logger.error(f"Failed to record search activity: {e}")

    def _on_search_failed(self, message: str, generation: int = 0) -> None:
        if generation and generation != self._search_generation:
            logger.info("Ignoring stale search failure (generation %s)", generation)
            return
        self._timeout_timer.stop()
        self._worker = None
        self._set_state(SearchState.ERROR)
        self.status_message.emit(f"Search failed: {message}")
        self.search_error.emit(message)

    @Slot()
    def _on_search_timeout(self) -> None:
        if self._state != SearchState.SEARCHING:
            return
        logger.error(
            "Search timed out after %ss for %s",
            self._search_timeout_ms / 1000.0,
            self._last_query_path,
        )
        self._search_generation += 1
        worker = self._worker
        self._worker = None
        if worker is not None and worker.isRunning():
            try:
                worker.requestInterruption()
            except Exception:
                pass
        self._set_state(SearchState.ERROR)
        message = (
            "Search is taking too long and was stopped.\n\n"
            "Common fixes:\n"
            "• Wait for Indexing to finish, then try again\n"
            "• Use Auto Crop & Search on room photos\n"
            "• On Mac, restart the app if search stays stuck\n"
            "• Confirm tiles appear under Index before searching"
        )
        self.status_message.emit("Search timed out.")
        self.search_error.emit(message)

    @Slot(float)
    def _on_search_timed(self, elapsed_seconds: float) -> None:
        self._last_elapsed_seconds = elapsed_seconds
        logger.info(f"Search for '{self._last_query_path}' completed in {elapsed_seconds:.3f}s.")

    @Slot()
    def load_search_history(self, limit: int = 10) -> None:
        if self._history_repo is None:
            self.search_history_updated.emit([])
            return

        try:
            entries = self._history_repo.get_recent_searches(limit=limit)
            self.search_history_updated.emit(entries)
        except Exception as e:
            logger.error(f"Failed to load search history: {e}")
            self.search_history_updated.emit([])

    @Slot(str)
    def repeat_search(self, query_image_path: str) -> None:
        if not Path(query_image_path).exists():
            self.search_error.emit(
                f"That search's original image no longer exists:\n{query_image_path}"
            )
            return
        self.search_by_image(query_image_path)
