"""
Folder monitoring use case for TileVision AI.

Utilizes the Watchdog library to monitor directories for new/updated images,
handling delay buffering to avoid reading half-copied files.
"""

import logging
import os
from pathlib import Path
import time
from typing import Callable, List, Optional, Set

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
except ImportError:
    # Mock fallback for environment checks
    Observer = None
    FileSystemEventHandler = object
    FileSystemEvent = object

from src.core.use_cases.index_images import IndexImagesUseCase
from src.utils.image_utils import validate_image, SUPPORTED_IMAGE_EXTENSIONS

logger = logging.getLogger("tilevision.core.use_cases.monitor_folder")


class TileImageEventHandler(FileSystemEventHandler):
    """
    Listens for filesystem events and triggers indexing when new files are created/moved.
    """

    def __init__(
        self,
        indexing_use_case: IndexImagesUseCase,
        on_file_indexed_callback: Optional[Callable[[str, bool, str], None]] = None,
        settle_delay_seconds: float = 2.0,
    ) -> None:
        """
        Initialize the event handler.

        Args:
            indexing_use_case: Use case to index new image files.
            on_file_indexed_callback: Callback triggered after indexing,
                                      receiving (file_path, success, error_message).
            settle_delay_seconds: Seconds to wait for file writing to finish before reading.
        """
        self._use_case = indexing_use_case
        self._on_indexed = on_file_indexed_callback
        self._settle_delay = settle_delay_seconds
        self._supported_extensions: Set[str] = set(SUPPORTED_IMAGE_EXTENSIONS)

    def _is_supported(self, file_path: Path) -> bool:
        """Check if file extension is supported."""
        return file_path.suffix.lower() in self._supported_extensions

    def _process_file(self, file_path_str: str) -> None:
        """Verify file write completion and run indexing."""
        file_path = Path(file_path_str).resolve()
        if not self._is_supported(file_path):
            return

        logger.info(f"Detected filesystem activity for: {file_path.name}")
        
        # Settle check: wait for the file to be completely copied/written.
        # Check size stability over the settle delay window.
        try:
            last_size = -1
            retries = 5
            for attempt in range(retries):
                time.sleep(self._settle_delay / retries)
                if not file_path.exists():
                    logger.warning(f"File vanished during stabilization: {file_path}")
                    return
                current_size = file_path.stat().st_size
                if current_size == last_size and current_size > 0:
                    # File size stabilized, check if valid image
                    if validate_image(file_path):
                        break
                last_size = current_size
            else:
                logger.warning(f"File did not stabilize or is invalid: {file_path}")
                if self._on_indexed:
                    self._on_indexed(str(file_path), False, "File write timeout or invalid format.")
                return
        except OSError as e:
            logger.error(f"Failed to access file stat during stabilization: {e}")
            if self._on_indexed:
                self._on_indexed(str(file_path), False, f"OS error: {e}")
            return

        # Trigger indexing usecase
        try:
            logger.info(f"Indexing new file: {file_path.name}")
            # index_single_file() persists the FAISS index to disk itself
            # (persist=True by default), so no extra save call is needed here.
            db_id = self._use_case.index_single_file(file_path)

            logger.info(f"Successfully indexed background file. ID: {db_id}")
            if self._on_indexed:
                self._on_indexed(str(file_path), True, "")
        except Exception as e:
            logger.error(f"Background indexing failed for file {file_path}: {e}")
            if self._on_indexed:
                self._on_indexed(str(file_path), False, str(e))

    def on_created(self, event: FileSystemEvent) -> None:
        """Watchdog handler for file creation."""
        if not event.is_directory:
            self._process_file(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        """Watchdog handler for file moves/renames."""
        if not event.is_directory:
            self._process_file(event.dest_path)


class FolderMonitorController:
    """
    Manages starting and stopping Watchdog observers across multiple paths.
    """

    def __init__(
        self,
        indexing_use_case: IndexImagesUseCase,
        on_file_indexed_callback: Optional[Callable[[str, bool, str], None]] = None,
    ) -> None:
        """
        Initialize the controller.

        Args:
            indexing_use_case: IndexImagesUseCase dependency.
            on_file_indexed_callback: Signal callback when an auto-index event finishes.
        """
        self._use_case = indexing_use_case
        self._on_indexed = on_file_indexed_callback
        self._observer = None
        self._active_watches = []

    def start_monitoring(self, folders: List[str]) -> None:
        """
        Begin monitoring the designated list of folder paths.

        Args:
            folders: List of absolute directory path strings.
        """
        if Observer is None:
            logger.critical("watchdog package not installed! Cannot monitor folders.")
            raise ImportError("watchdog package is required for FolderMonitorController.")

        if self._observer is not None:
            self.stop_monitoring()

        self._observer = Observer()
        self._active_watches = []
        
        handler = TileImageEventHandler(
            indexing_use_case=self._use_case,
            on_file_indexed_callback=self._on_indexed
        )

        for folder_str in folders:
            folder_path = Path(folder_str).resolve()
            if not folder_path.exists() or not folder_path.is_dir():
                logger.warning(f"Monitored folder path does not exist or is not a directory: {folder_path}")
                continue

            logger.info(f"Registering watcher for directory: {folder_path}")
            try:
                watch = self._observer.schedule(handler, path=str(folder_path), recursive=True)
                self._active_watches.append(watch)
            except Exception as e:
                logger.error(f"Failed to start watchdog watcher for path {folder_path}: {e}")

        if self._active_watches:
            self._observer.start()
            logger.info("Watchdog folder observer thread started successfully.")
        else:
            logger.warning("No valid folders monitored. Watchdog was not started.")

    def stop_monitoring(self) -> None:
        """Stop all watches and terminate the observer thread."""
        if self._observer is not None:
            logger.info("Stopping folder monitor observer thread...")
            self._observer.stop()
            self._observer.join()
            self._observer = None
            self._active_watches = []
            logger.info("Folder monitor observer stopped.")
