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

    # (indexed_count, skipped_count, total_count)
    indexing_completed = Signal(int, int, int)

    # (error_message: str)
    error_occurred = Signal(str)

    # (folder_path: str)
    folder_selected = Signal(str)

    # (message: str) — informational status line
    status_message = Signal(str)

    def __init__(self, use_case: IndexImagesUseCase, parent: Optional[QObject] = None) -> None:
        """
        Initialize the IndexingViewModel.

        Args:
            use_case: Fully configured IndexImagesUseCase instance.
            parent: Optional Qt parent object.
        """
        super().__init__(parent)
        self._use_case = use_case
        self._worker: Optional[IndexingWorker] = None
        self._state: str = IndexingState.IDLE
        self._selected_folder: Optional[Path] = None
        self._total_count: int = 0
        self._processed_count: int = 0

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
        """True if no indexing job is active."""
        return self._state in (IndexingState.IDLE, IndexingState.FINISHED, IndexingState.CANCELLED)

    @property
    def is_running(self) -> bool:
        """True if indexing is actively running."""
        return self._state == IndexingState.RUNNING

    @property
    def is_paused(self) -> bool:
        """True if the indexing thread is currently paused."""
        return self._state == IndexingState.PAUSED

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
        """
        if self._worker and self._state in (IndexingState.RUNNING, IndexingState.PAUSED):
            logger.warning("Cancelling indexing worker by user request.")
            self._worker.cancel()
            self._set_state(IndexingState.CANCELLED)
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

    @Slot(int, int, bool)
    def _on_indexing_finished(self, indexed: int, skipped: int, completed: bool) -> None:
        """
        Handle the indexing_finished signal from the worker.

        Args:
            indexed: Number of new tiles successfully indexed.
            skipped: Number of unchanged tiles skipped.
            completed: True if indexing completed fully without cancellation.
        """
        if completed:
            self._set_state(IndexingState.FINISHED)
            summary = (
                f"Indexing complete. "
                f"Indexed: {indexed:,} new tiles, "
                f"Skipped: {skipped:,} unchanged."
            )
            self.status_message.emit(summary)
            logger.info(summary)
        else:
            # Worker stopped due to cancellation
            if self._state != IndexingState.CANCELLED:
                self._set_state(IndexingState.CANCELLED)
            self.status_message.emit(
                f"Indexing cancelled. Indexed {indexed:,} tiles before stopping."
            )

        self.indexing_completed.emit(indexed, skipped, self._total_count)
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
