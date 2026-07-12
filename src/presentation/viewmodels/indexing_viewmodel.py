"""
Indexing ViewModel for TileVision AI.

Mediates between the IndexingWorker (background QThread) and the IndexingView UI.
Manages state transitions (Idle → Running → Paused → Finished/Cancelled),
exposes formatted progress strings, ETA, and emits Qt Signals to update the view.

Design Decision:
    The ViewModel owns the lifecycle of IndexingWorker instances.
    It creates a new worker each time indexing is started to avoid QThread reuse issues.
    All UI-facing formatting (ETA strings, percentage) is done here, keeping the View thin.
"""

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal, Slot

from src.core.use_cases.index_images import IndexImagesUseCase
from src.presentation.workers.indexing_worker import IndexingWorker

logger = logging.getLogger("tilevision.presentation.viewmodels.indexing_viewmodel")


class IndexingState:
    """Enumeration of valid indexing lifecycle states."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    ERROR = "error"


class IndexingViewModel(QObject):
    """
    ViewModel for the Folder Indexing feature.

    Exposes Qt Signals that the IndexingView connects to.
    Provides slots for UI user actions (start, pause, resume, cancel).
    """

    # ── Signals emitted TO the View ──────────────────────────────────────────

    # (processed_count, total_count, percent, current_filename, eta_display_string)
    progress_changed = Signal(int, int, int, str, str)

    # (state: str) — one of IndexingState constants
    state_changed = Signal(str)

    # ScanResult — the full new/modified/deleted/skipped breakdown
    # (Task 2: Smart Re-index), not just a flat total.
    indexing_completed = Signal(object)

    # (error_message: str)
    error_occurred = Signal(str)

    # (folder_path: str)
    folder_selected = Signal(str)

    # (message: str) — informational status line
    status_message = Signal(str)

    # IndexedFolderState — emitted at startup (Task 1: Persistent Indexed
    # Folder) if a previously-indexed folder was found, so the View can
    # restore "Folder: X / Indexed Images: N / Status: Ready / Last
    # Indexed: ..." without requiring the user to re-select or re-scan.
    persisted_folder_loaded = Signal(object)

    def __init__(
        self, use_case: IndexImagesUseCase, parent: Optional[QObject] = None,
        activity_log_repository=None,
    ) -> None:
        """
        Initialize the IndexingViewModel.

        Args:
            use_case: Fully configured IndexImagesUseCase instance.
            parent: Optional Qt parent object.
            activity_log_repository: Optional repository for the
                Dashboard's Recent Activity feed (Task A). If omitted,
                indexing events simply aren't recorded there.
        """
        super().__init__(parent)
        self._use_case = use_case
        self._worker: Optional[IndexingWorker] = None
        self._state: str = IndexingState.IDLE
        self._selected_folder: Optional[Path] = None
        self._total_count: int = 0
        self._processed_count: int = 0
        self._activity_repo = activity_log_repository

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        """Current indexing lifecycle state."""
        return self._state

    @property
    def selected_folder(self) -> Optional[Path]:
        """Currently selected folder path."""
        return self._selected_folder

    @property
    def is_idle(self) -> bool:
        """
        True if no indexing job is active and a new one can safely be started.

        CANCELLING is deliberately NOT included here: the background thread
        is still cooperatively finishing its current file at that point, and
        allowing a new job to start before it truly stops would let the old
        worker's delayed completion signal wipe out the reference to the new
        one (see cancel_indexing()).
        """
        return self._state in (IndexingState.IDLE, IndexingState.FINISHED, IndexingState.CANCELLED)

    @property
    def is_running(self) -> bool:
        """True if indexing is actively running."""
        return self._state == IndexingState.RUNNING

    @property
    def is_paused(self) -> bool:
        """True if the indexing thread is currently paused."""
        return self._state == IndexingState.PAUSED

    @Slot()
    def load_persisted_folder_state(self) -> None:
        """
        Restore the most recently indexed folder from persistent storage
        (Task 1: Persistent Indexed Folder), if one exists. Intended to be
        called once at startup (after the View has connected its signals),
        so the Index page shows "Folder: X / Indexed Images: N / Status:
        Ready / Last Indexed: ..." immediately, without the user needing
        to re-select or re-scan a folder they already indexed in a
        previous session.

        No-op (does not emit anything) if this use case wasn't configured
        with a folder repository, or if no folder has ever been indexed.
        """
        get_status = getattr(self._use_case, "get_last_indexed_folder_status", None)
        if get_status is None:
            logger.debug(
                "Use case has no get_last_indexed_folder_status() — skipping folder restoration."
            )
            return

        status = get_status()
        if status is None:
            logger.info("No persisted folder state found — Index page starts empty.")
            return

        self._selected_folder = Path(status.folder_path)
        self._set_state(IndexingState.FINISHED)
        logger.info(
            f"Restored persisted folder state: {status.folder_path} "
            f"({status.indexed_image_count} images, last indexed {status.last_indexed_at})"
        )
        self.persisted_folder_loaded.emit(status)

    # ── Public Slots (called by the View) ────────────────────────────────────

    @Slot(str)
    def set_folder(self, folder_path: str) -> None:
        """
        Set the folder selected by the user via the folder picker dialog.

        Args:
            folder_path: Absolute path string to the target directory.
        """
        resolved = Path(folder_path).resolve()
        if not resolved.exists() or not resolved.is_dir():
            logger.warning(f"Invalid folder selected: {folder_path}")
            self.error_occurred.emit(f"Selected path is not a valid directory:\n{folder_path}")
            return

        self._selected_folder = resolved
        logger.info(f"Folder selected: {resolved}")
        self.folder_selected.emit(str(resolved))
        self.status_message.emit(f"Folder ready: {resolved.name}")

    @Slot()
    def start_indexing(self) -> None:
        """
        Begin the folder indexing process.

        Creates a new IndexingWorker, connects signals, and starts the background thread.
        Validates that a folder has been selected before proceeding.
        """
        if self._selected_folder is None:
            self.error_occurred.emit("Please select a folder before starting indexing.")
            return

        if not self.is_idle:
            logger.warning(f"Start requested while in state '{self._state}'. Ignoring.")
            return

        logger.info(f"Starting indexing for folder: {self._selected_folder}")

        # Create a fresh worker for each run (QThread reuse is not safe in PySide6)
        self._worker = IndexingWorker(
            use_case=self._use_case,
            directory_path=self._selected_folder,
        )

        # Connect worker signals to our handler slots
        self._worker.progress_updated.connect(self._on_progress_updated)
        self._worker.indexing_finished.connect(self._on_indexing_finished)
        self._worker.indexing_error.connect(self._on_indexing_error)

        # Cleanup the worker object when the thread naturally finishes
        self._worker.finished.connect(self._worker.deleteLater)

        self._set_state(IndexingState.RUNNING)
        self.status_message.emit(f"Scanning: {self._selected_folder.name}...")
        self._worker.start()

    @Slot()
    def pause_indexing(self) -> None:
        """
        Cooperatively pause the running indexing worker.
        The worker will finish processing the current file before pausing.
        """
        if self._worker and self._state == IndexingState.RUNNING:
            logger.info("Pausing indexing worker.")
            self._worker.pause()
            self._set_state(IndexingState.PAUSED)
            self.status_message.emit("Indexing paused.")
        else:
            logger.warning(f"Pause called in invalid state: {self._state}")

    @Slot()
    def resume_indexing(self) -> None:
        """
        Resume a previously paused indexing worker.
        """
        if self._worker and self._state == IndexingState.PAUSED:
            logger.info("Resuming indexing worker.")
            self._worker.resume()
            self._set_state(IndexingState.RUNNING)
            self.status_message.emit("Indexing resumed...")
        else:
            logger.warning(f"Resume called in invalid state: {self._state}")

    @Slot()
    def cancel_indexing(self) -> None:
        """
        Cooperatively cancel the running or paused indexing operation.
        The worker will stop after the current file completes.

        Transitions to CANCELLING (not CANCELLED) immediately: the background
        thread is still cooperatively finishing its current file and hasn't
        actually stopped yet. CANCELLING is intentionally excluded from
        is_idle so Start/Browse stay disabled and no second worker can be
        started until the real indexing_finished signal arrives — starting
        one earlier would let the still-running old worker's completion
        signal null out the reference to the new worker.
        """
        if self._worker and self._state in (IndexingState.RUNNING, IndexingState.PAUSED):
            logger.warning("Cancelling indexing worker by user request.")
            self._worker.cancel()
            self._set_state(IndexingState.CANCELLING)
            self.status_message.emit("Cancelling indexing... Please wait.")
        else:
            logger.warning(f"Cancel called in invalid state: {self._state}")

    # ── Private Slots (connected to worker signals) ───────────────────────────

    @Slot(int, int, str, float)
    def _on_progress_updated(
        self, processed: int, total: int, filename: str, eta_seconds: float
    ) -> None:
        """
        Handle progress signal from the IndexingWorker.
        Formats ETA string and emits progress_changed signal.

        Args:
            processed: Number of files processed so far.
            total: Total file count to process.
            filename: Name of the currently processed file.
            eta_seconds: Estimated remaining time in seconds.
        """
        self._processed_count = processed
        self._total_count = total

        percent = int((processed / total) * 100) if total > 0 else 0
        eta_string = self._format_eta(eta_seconds)

        self.progress_changed.emit(processed, total, percent, filename, eta_string)

    @Slot(object)
    def _on_indexing_finished(self, result) -> None:
        """
        Handle the indexing_finished signal from the worker.

        Args:
            result: The ScanResult from IndexImagesUseCase.scan_and_index_directory().
        """
        if result.is_completed:
            self._set_state(IndexingState.FINISHED)

            if not result.has_any_changes:
                # Task 2: nothing to do — show a clear, reassuring message
                # rather than a summary line of all zeros.
                summary = "Everything is already indexed."
            else:
                parts = []
                if result.new_count:
                    parts.append(f"{result.new_count:,} new")
                if result.modified_count:
                    parts.append(f"{result.modified_count:,} modified")
                if result.deleted_count:
                    parts.append(f"{result.deleted_count:,} removed")
                if result.skipped_count:
                    parts.append(f"{result.skipped_count:,} unchanged")
                summary = "Indexing complete — " + ", ".join(parts) + "."
                if result.time_saved_seconds >= 1:
                    summary += f" Saved ~{self._format_eta(result.time_saved_seconds)} by skipping unchanged files."

            self.status_message.emit(summary)
            logger.info(summary)
        else:
            # Worker stopped due to cancellation (covers both a user-initiated
            # cancel, now arriving from CANCELLING, and any other early-exit
            # path that reports completed=False).
            self._set_state(IndexingState.CANCELLED)
            self.status_message.emit(
                f"Indexing cancelled. Indexed {result.indexed_count:,} tiles before stopping."
            )

        if self._activity_repo is not None and result.is_completed and result.has_any_changes:
            try:
                folder_name = self._selected_folder.name if self._selected_folder else "folder"
                parts = []
                if result.new_count:
                    parts.append(f"{result.new_count} new")
                if result.modified_count:
                    parts.append(f"{result.modified_count} modified")
                if result.deleted_count:
                    parts.append(f"{result.deleted_count} removed")
                self._activity_repo.record_activity(
                    "index", f"Indexed '{folder_name}' — {', '.join(parts)}"
                )
            except Exception as e:
                logger.error(f"Failed to record indexing activity: {e}")

        self.indexing_completed.emit(result)
        self._worker = None

    @Slot(str)
    def _on_indexing_error(self, error_message: str) -> None:
        """
        Handle the indexing_error signal from the worker.

        Args:
            error_message: Human-readable error description from the worker.
        """
        logger.error(f"Indexing worker reported a critical error: {error_message}")
        self._set_state(IndexingState.ERROR)
        self.error_occurred.emit(f"Indexing failed with error:\n{error_message}")
        self.status_message.emit("Indexing failed. See error details.")
        self._worker = None

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _set_state(self, new_state: str) -> None:
        """
        Update the internal state and emit state_changed signal.

        Args:
            new_state: One of the IndexingState constants.
        """
        if self._state != new_state:
            logger.debug(f"State transition: {self._state} → {new_state}")
            self._state = new_state
            self.state_changed.emit(new_state)

    @staticmethod
    def _format_eta(eta_seconds: float) -> str:
        """
        Convert remaining seconds to a human-readable countdown string.

        Args:
            eta_seconds: Number of seconds estimated remaining.

        Returns:
            Formatted string like "2m 34s", "45s", or "--" for unknown.
        """
        if eta_seconds <= 0:
            return "--"

        total_secs = int(eta_seconds)
        hours = total_secs // 3600
        minutes = (total_secs % 3600) // 60
        seconds = total_secs % 60

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
