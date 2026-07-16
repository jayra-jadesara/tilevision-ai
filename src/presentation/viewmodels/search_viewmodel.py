"""
Search ViewModel module for TileVision AI.

Manages the state of a visual similarity search: accepting a query image
(drag-and-drop or browse), running it on a background worker thread, and
exposing results/status to the SearchView through Qt signals.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, Signal, Slot

from src.core.models import SearchResult, SearchHistoryEntry
from src.core.use_cases.search_tiles import SearchTilesUseCase
from src.data.repository_interface import ISearchHistoryRepository, IActivityLogRepository
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

    filters_available = Signal(dict)  # Dict[str, List[str]] — for populating dropdowns

    # (result_count, elapsed_seconds) — Task C: Search UX (elapsed time + results count display)
    search_stats_ready = Signal(int, float)

    # List[SearchHistoryEntry] — Task C: Search UX (search history panel)
    search_history_updated = Signal(list)

    def __init__(
        self,
        use_case: SearchTilesUseCase,
        default_top_k: int = 20,
        search_history_repository: Optional[ISearchHistoryRepository] = None,
        activity_log_repository: Optional[IActivityLogRepository] = None,
    ) -> None:
        """
        Initialize the SearchViewModel.

        Args:
            use_case: Fully configured SearchTilesUseCase.
            default_top_k: Default maximum number of results to request.
            search_history_repository: Optional repository for recording/
                retrieving search history (Task C). If omitted, searches
                simply aren't recorded — kept optional for backward
                compatibility with any existing construction sites/tests.
            activity_log_repository: Optional repository for the
                Dashboard's Recent Activity feed (Task A).
        """
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

    @Slot()
    def load_filter_options(self) -> None:
        """
        Fetch the available filter values (brand/category/color/size) from
        the catalog and emit them via filters_available for the View to
        populate its dropdowns. Safe to call even on an empty catalog
        (returns empty lists per field rather than erroring).
        """
        try:
            options = self._use_case.get_filter_options()
            self.filters_available.emit(options)
        except Exception as e:
            logger.error(f"Failed to load filter options: {e}")
            self.filters_available.emit({})

    @Slot(str, str)
    def set_filter(self, field: str, value: str) -> None:
        """
        Update a single filter and, if a query image is already active,
        automatically re-run the search with the new filter applied.

        Args:
            field: One of "brand", "category", "color", "size".
            value: The required value, or "" / "Any" to clear that filter.
        """
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

        try:
            health = self._use_case.get_index_health()
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

        self._set_state(SearchState.SEARCHING)
        self.status_message.emit(f"Searching for tiles similar to '{path.name}'...")

        self._worker = SearchWorker(self._use_case, str(path), self._top_k, self._active_filters)
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

    @Slot(str)
    def _on_search_failed(self, message: str) -> None:
        self._worker = None
        self._set_state(SearchState.ERROR)
        self.status_message.emit(f"Search failed: {message}")
        self.search_error.emit(message)

    @Slot(float)
    def _on_search_timed(self, elapsed_seconds: float) -> None:
        self._last_elapsed_seconds = elapsed_seconds
        logger.info(f"Search for '{self._last_query_path}' completed in {elapsed_seconds:.3f}s.")

    # ── Search History (Task C) ─────────────────────────────────────────

    @Slot()
    def load_search_history(self, limit: int = 10) -> None:
        """
        Fetch recent searches and emit them via search_history_updated for
        the View to render as a clickable history panel.

        Args:
            limit: Maximum number of history entries to retrieve.
        """
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
        """
        Re-run a past search from history (Task C), if the original query
        image still exists on disk.

        Args:
            query_image_path: Absolute path from a SearchHistoryEntry.
        """
        if not Path(query_image_path).exists():
            self.search_error.emit(
                f"That search's original image no longer exists:\n{query_image_path}"
            )
            return
        self.search_by_image(query_image_path)
