"""
Search ViewModel module for TileVision AI.

Manages the state of a visual similarity search: accepting a query image
(drag-and-drop or browse), running it on a background worker thread, and
exposing results/status to the SearchView through Qt signals.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional

from PySide6.QtCore import QObject, QTimer, Qt, Signal, Slot

from src.ai.inference_guard import (
    InferenceBusyError,
    begin_search_priority,
    end_search_priority,
)
from src.core.models import SearchResult, SearchHistoryEntry
from src.core.use_cases.search_tiles import SearchTilesUseCase
from src.data.repository_interface import ISearchHistoryRepository, IActivityLogRepository
from src.presentation.workers.search_worker import SearchWorker

logger = logging.getLogger("tilevision.presentation.viewmodels.search_viewmodel")

# Soft status reminder only — do NOT abort a working search on Mac CPU.
# Hard abort caused false "Search took too long" errors while DINOv2 was still running.
# Set search_timeout_ms > 0 only for tests that exercise the abort path.
_SEARCH_TIMEOUT_MS = 0  # 0 = never auto-abort for wall-clock timeout
_SEARCH_STATUS_HINT_MS = 45_000
# Stall detector: abort only when heartbeats stop for this long.
# Heartbeats fire every ~5s from SearchWorker while the QThread is alive.
_SEARCH_STALL_MS = 90_000


def _default_search_timeout_ms() -> int:
    return _SEARCH_TIMEOUT_MS


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
        search_timeout_ms: Optional[int] = None,
        on_search_busy_changed: Optional[Callable[[bool], None]] = None,
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
        timeout = (
            _default_search_timeout_ms()
            if search_timeout_ms is None
            else int(search_timeout_ms)
        )
        # 0 disables the abort timer permanently (production default).
        self._search_timeout_ms = max(0, timeout)
        self._timeout_timer = QTimer(self)
        self._timeout_timer.setSingleShot(True)
        self._timeout_timer.timeout.connect(self._on_search_timeout)
        self._status_hint_timer = QTimer(self)
        self._status_hint_timer.setSingleShot(True)
        self._status_hint_timer.timeout.connect(self._on_search_status_hint)
        # Heartbeat-based stall monitor (not an arbitrary search timeout).
        self._stall_timer = QTimer(self)
        self._stall_timer.setSingleShot(True)
        self._stall_timer.timeout.connect(self._on_search_stall)
        self._on_search_busy_changed = on_search_busy_changed
        self._search_priority_held = False
        self._pending_query_path: Optional[str] = None

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
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            self._set_state(SearchState.ERROR)
            self.search_error.emit(f"Selected file does not exist: {image_path}")
            self.status_message.emit("Search failed: file not found.")
            logger.warning("[SEARCH] Drop/path rejected — file missing: %s", image_path)
            return

        if self._state == SearchState.SEARCHING:
            # Do not drop the user's second image — run it when the current one finishes.
            self._pending_query_path = str(path)
            logger.warning(
                "[SEARCH] Search already in progress; queued next query: %s",
                path.name,
            )
            self.status_message.emit(
                f"Search in progress — will search '{path.name}' next…"
            )
            return

        # Claim priority BEFORE FAISS health probes so indexing yields the lock
        # instead of the UI mistaking a busy lock for an empty index.
        self._claim_search_priority()

        try:
            health = self._use_case.get_index_health()
            indexed = int(getattr(health, "indexed_count", 0) or 0)
            if indexed <= 0:
                self._release_search_priority()
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
                self._release_search_priority()
                self._set_state(SearchState.ERROR)
                self.search_error.emit(
                    "Indexed features are outdated. "
                    f"{health.stale_count} of {health.indexed_count} tiles "
                    "need re-indexing.\n\n"
                    "Go to Settings → Rebuild FAISS Index, then search again."
                )
                self.status_message.emit("Search blocked: index is outdated.")
                return
            try:
                searchable = self._use_case.get_searchable_count()
            except InferenceBusyError:
                logger.warning(
                    "[SEARCH] FAISS busy during health check — continuing; worker will wait."
                )
                searchable = None
            if searchable is not None and searchable <= 0:
                self._release_search_priority()
                self._set_state(SearchState.ERROR)
                self.search_error.emit(
                    "Catalog metadata is present but the searchable vector index is empty.\n\n"
                    "Go to Settings → Rebuild FAISS Index, wait until it finishes, "
                    "then drop your image again."
                )
                self.status_message.emit("Search blocked: vector index is empty.")
                return
        except Exception as exc:
            # Non-fatal: worker will re-validate and surface a clear error.
            logger.warning(
                "[SEARCH] Could not verify feature index health (continuing): %s",
                exc,
            )

        self._last_query_path = str(path)
        self.query_image_selected.emit(str(path))

        self._search_generation += 1
        generation = self._search_generation

        self._set_state(SearchState.SEARCHING)
        self.status_message.emit(f"Searching for tiles similar to '{path.name}'...")
        logger.info("[SEARCH] Starting worker for %s (generation=%s)", path.name, generation)

        self._worker = SearchWorker(self._use_case, str(path), self._top_k, self._active_filters)
        self._worker.search_completed.connect(
            lambda results, gen=generation: self._on_search_completed(results, gen)
        )
        self._worker.search_failed.connect(
            lambda message, gen=generation: self._on_search_failed(message, gen)
        )
        self._worker.search_timed.connect(self._on_search_timed)
        self._worker.search_progress.connect(self._on_search_progress)
        self._worker.search_heartbeat.connect(
            self._on_search_heartbeat,
            Qt.ConnectionType.QueuedConnection,
        )
        worker = self._worker
        worker.finished.connect(worker.deleteLater)
        if self._search_timeout_ms > 0:
            self._timeout_timer.start(self._search_timeout_ms)
        self._status_hint_timer.start(_SEARCH_STATUS_HINT_MS)
        self._stall_timer.start(_SEARCH_STALL_MS)
        self._worker.start()

    def _start_pending_search_if_any(self) -> None:
        pending = self._pending_query_path
        self._pending_query_path = None
        if pending and self._state != SearchState.SEARCHING:
            logger.info("[SEARCH] Starting queued query: %s", pending)
            self.search_by_image(pending)

    @Slot()
    def clear_results(self) -> None:
        self._timeout_timer.stop()
        self._status_hint_timer.stop()
        self._stall_timer.stop()
        self._pending_query_path = None
        self._search_generation += 1
        worker = self._worker
        self._worker = None
        if worker is not None and worker.isRunning():
            try:
                worker.requestInterruption()
            except Exception:
                pass
        self._release_search_priority()
        self._last_results = []
        self._last_query_path = None
        self._set_state(SearchState.IDLE)
        self.results_ready.emit([])
        self.status_message.emit("Ready. Drag an image or click Browse to search.")

    def _set_state(self, new_state: str) -> None:
        if self._state != new_state:
            self._state = new_state
            self.state_changed.emit(new_state)

    def _claim_search_priority(self) -> None:
        if self._search_priority_held:
            return
        begin_search_priority()
        self._search_priority_held = True
        if self._on_search_busy_changed is not None:
            try:
                self._on_search_busy_changed(True)
            except Exception as exc:
                logger.warning("on_search_busy_changed(True) failed: %s", exc)

    def _release_search_priority(self) -> None:
        if not self._search_priority_held:
            return
        end_search_priority()
        self._search_priority_held = False
        if self._on_search_busy_changed is not None:
            try:
                self._on_search_busy_changed(False)
            except Exception as exc:
                logger.warning("on_search_busy_changed(False) failed: %s", exc)

    def _on_search_completed(self, results: List[SearchResult], generation: int = 0) -> None:
        if generation and generation != self._search_generation:
            logger.info("Ignoring stale search completion (generation %s)", generation)
            return
        self._timeout_timer.stop()
        self._status_hint_timer.stop()
        self._stall_timer.stop()
        self._release_search_priority()
        self._last_results = results
        self._worker = None

        if results:
            self._set_state(SearchState.RESULTS)
            self.status_message.emit(f"Found {len(results)} similar tile(s).")
            logger.info("[SEARCH] Results displayed: %d", len(results))
        else:
            self._set_state(SearchState.NO_RESULTS)
            self.status_message.emit(
                "No similar tiles found in the indexed catalog. "
                "Try another photo, Auto Crop, or check that your catalogue is indexed."
            )
            logger.warning("[SEARCH] Completed with 0 results for %s", self._last_query_path)

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

        self._start_pending_search_if_any()

    def _on_search_failed(self, message: str, generation: int = 0) -> None:
        if generation and generation != self._search_generation:
            logger.info("Ignoring stale search failure (generation %s)", generation)
            return
        self._timeout_timer.stop()
        self._status_hint_timer.stop()
        self._stall_timer.stop()
        self._release_search_priority()
        self._worker = None

        # User-initiated cancel / clear — do not show a scary error dialog.
        if message == "Search cancelled":
            if self._state == SearchState.SEARCHING:
                self._set_state(SearchState.IDLE)
                self.status_message.emit("Search cancelled.")
            logger.info("[SEARCH] Worker reported cancel for generation %s", generation)
            self._start_pending_search_if_any()
            return

        self._set_state(SearchState.ERROR)
        self.status_message.emit(f"Search failed: {message}")
        self.search_error.emit(message)
        logger.error("[SEARCH] Failed: %s", message)
        self._start_pending_search_if_any()

    @Slot(str)
    def _on_search_progress(self, message: str) -> None:
        if self._state != SearchState.SEARCHING:
            return
        # Progress also counts as liveness.
        self._stall_timer.start(_SEARCH_STALL_MS)
        self.status_message.emit(message)

    @Slot()
    def _on_search_heartbeat(self) -> None:
        """Reset stall watchdog — worker is alive even during long DINOv2."""
        if self._state != SearchState.SEARCHING:
            return
        self._stall_timer.start(_SEARCH_STALL_MS)

    @Slot()
    def _on_search_status_hint(self) -> None:
        """Reassure the user — do not abort. Mac CPU can take a minute."""
        if self._state != SearchState.SEARCHING:
            return
        self.status_message.emit(
            "Still searching… AI matching can take up to a minute on this computer. "
            "Please wait — results will appear here."
        )

    @Slot()
    def _on_search_stall(self) -> None:
        """
        Abort only when heartbeats stop (worker appears wedged).

        A slow but heartbeating DINOv2 forward on Mac Intel must finish.
        """
        if self._state != SearchState.SEARCHING:
            return
        logger.error(
            "Search stalled (no heartbeat for %ss) for %s",
            _SEARCH_STALL_MS / 1000.0,
            self._last_query_path,
        )
        self._timeout_timer.stop()
        self._status_hint_timer.stop()
        self._search_generation += 1
        worker = self._worker
        self._worker = None
        if worker is not None and worker.isRunning():
            try:
                worker.requestInterruption()
            except Exception:
                pass
        self._release_search_priority()
        self._set_state(SearchState.ERROR)
        message = (
            "Search stopped because the AI worker stopped responding.\n\n"
            "Pause Indexing, restart the app, then try again.\n\n"
            "Also try Auto Crop & Search on a smaller crop of the tile."
        )
        self.status_message.emit("Search stopped — worker unresponsive.")
        self.search_error.emit(message)

    @Slot()
    def _on_search_timeout(self) -> None:
        """Optional hard abort (disabled in production; used by unit tests only)."""
        if self._search_timeout_ms <= 0:
            return
        if self._state != SearchState.SEARCHING:
            return
        logger.error(
            "Search timed out after %ss for %s",
            self._search_timeout_ms / 1000.0,
            self._last_query_path,
        )
        self._status_hint_timer.stop()
        self._stall_timer.stop()
        self._search_generation += 1
        worker = self._worker
        self._worker = None
        if worker is not None and worker.isRunning():
            try:
                worker.requestInterruption()
            except Exception:
                pass
        self._release_search_priority()
        self._set_state(SearchState.ERROR)
        message = (
            "Search took too long and was stopped.\n\n"
            "Pause Indexing, then drop the image again.\n\n"
            "Also try Auto Crop & Search, or restart the app."
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
