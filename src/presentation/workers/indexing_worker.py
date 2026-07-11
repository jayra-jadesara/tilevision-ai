"""
Indexing worker module for TileVision AI.

Implements a PySide6 QThread class to execute folder scanning and indexing
operations in the background, keeping the user interface completely responsive.
Supports pause, resume, and cancel signals.
"""

import logging
from pathlib import Path
import threading
from PySide6.QtCore import QThread, Signal

from src.core.use_cases.index_images import IndexImagesUseCase

logger = logging.getLogger("tilevision.presentation.workers.indexing_worker")


class IndexingWorker(QThread):
    """
    Background worker thread for indexing tile directories recursively.
    
    Exposes thread-safe Qt Signals to communicate progress, completion status,
    and errors to the presentation ViewModels/Views.
    """

    # Signal payload: (processed_count, total_count, current_filename, eta_seconds)
    progress_updated = Signal(int, int, str, float)
    
    # Signal payload: ScanResult — the full new/modified/deleted/skipped
    # breakdown (Task 2: Smart Re-index), not just a flat total.
    indexing_finished = Signal(object)
    
    # Signal payload: (error_message)
    indexing_error = Signal(str)

    def __init__(self, use_case: IndexImagesUseCase, directory_path: Path) -> None:
        """
        Initialize the indexing background worker.

        Args:
            use_case: Fully configured IndexImagesUseCase.
            directory_path: Directory path that needs to be scanned.
        """
        super().__init__()
        self._use_case = use_case
        self._directory_path = directory_path
        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()

    def run(self) -> None:
        """Execute the indexing process in the background thread."""
        logger.info(f"Indexing QThread started for folder: {self._directory_path}")
        self._cancel_event.clear()
        self._pause_event.clear()

        # Internal progress adapter to bridge standard Python callback to PySide6 Signal
        def _qt_progress_callback(processed: int, total: int, filename: str, eta: float) -> None:
            self.progress_updated.emit(processed, total, filename, eta)

        try:
            result = self._use_case.scan_and_index_directory(
                directory_path=self._directory_path,
                progress_callback=_qt_progress_callback,
                cancel_event=self._cancel_event,
                pause_event=self._pause_event,
            )
            
            logger.info(
                f"Indexing QThread finished. New: {result.new_count}, Modified: {result.modified_count}, "
                f"Deleted: {result.deleted_count}, Skipped: {result.skipped_count}, "
                f"Completed: {result.is_completed}"
            )
            self.indexing_finished.emit(result)
        except Exception as e:
            logger.error(f"Unexpected error in indexing background worker thread: {e}")
            self.indexing_error.emit(str(e))

    def pause(self) -> None:
        """Cooperately pause the indexing thread."""
        logger.info("Requesting pause for indexing worker.")
        self._pause_event.set()

    def resume(self) -> None:
        """Cooperately resume the indexing thread."""
        logger.info("Requesting resume for indexing worker.")
        self._pause_event.clear()

    def cancel(self) -> None:
        """Cooperately cancel the indexing thread."""
        logger.warning("Requesting cancel/termination for indexing worker.")
        self._cancel_event.set()
        # Also clear pause event to make sure it wakes up from wait and cancels immediately
        self._pause_event.clear()
