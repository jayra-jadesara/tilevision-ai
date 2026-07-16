"""
Folder monitoring use case for TileVision AI.

Utilizes the Watchdog library to monitor directories for new/updated/deleted
images, handling debouncing and settle buffering to avoid reading half-copied files.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
import time
from typing import Callable, Dict, List, Literal, Optional, Set

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    Observer = None
    FileSystemEventHandler = object
    FileSystemEvent = object
    WATCHDOG_AVAILABLE = False


def is_watchdog_available() -> bool:
    """Return True when the watchdog package is installed."""
    return WATCHDOG_AVAILABLE

from src.core.use_cases.index_images import IndexImagesUseCase
from src.utils.image_utils import validate_image, SUPPORTED_IMAGE_EXTENSIONS

logger = logging.getLogger("tilevision.core.use_cases.monitor_folder")

AutoIndexAction = Literal["indexed", "removed", "skipped", "failed"]
AutoIndexCallback = Callable[[str, AutoIndexAction, bool, str], None]


class TileImageEventHandler(FileSystemEventHandler):
    """
    Listens for filesystem events and triggers indexing when image files change.
    """

    def __init__(
        self,
        indexing_use_case: IndexImagesUseCase,
        on_file_indexed_callback: Optional[AutoIndexCallback] = None,
        settle_delay_seconds: float = 2.0,
        debounce_seconds: float = 1.0,
    ) -> None:
        self._use_case = indexing_use_case
        self._on_indexed = on_file_indexed_callback
        self._settle_delay = settle_delay_seconds
        self._debounce_seconds = debounce_seconds
        self._supported_extensions: Set[str] = set(SUPPORTED_IMAGE_EXTENSIONS)
        self._pending_timers: Dict[str, threading.Timer] = {}
        self._timer_lock = threading.Lock()

    def _is_supported(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self._supported_extensions

    def _schedule_process(self, file_path_str: str) -> None:
        key = str(Path(file_path_str).resolve())

        def _run() -> None:
            with self._timer_lock:
                self._pending_timers.pop(key, None)
            self._process_file(key)

        with self._timer_lock:
            existing = self._pending_timers.get(key)
            if existing is not None:
                existing.cancel()
            timer = threading.Timer(self._debounce_seconds, _run)
            timer.daemon = True
            self._pending_timers[key] = timer
            timer.start()

    def _notify(
        self,
        file_path: str,
        action: AutoIndexAction,
        success: bool,
        message: str = "",
    ) -> None:
        if self._on_indexed is not None:
            self._on_indexed(file_path, action, success, message)

    def _process_file(self, file_path_str: str) -> None:
        """Verify file write completion and run indexing."""
        file_path = Path(file_path_str).resolve()
        if not self._is_supported(file_path):
            return

        logger.info("Detected filesystem activity for: %s", file_path.name)

        try:
            last_size = -1
            retries = 5
            for _attempt in range(retries):
                time.sleep(self._settle_delay / retries)
                if not file_path.exists():
                    logger.warning("File vanished during stabilization: %s", file_path)
                    return
                current_size = file_path.stat().st_size
                if current_size == last_size and current_size > 0:
                    if validate_image(file_path):
                        break
                last_size = current_size
            else:
                logger.warning("File did not stabilize or is invalid: %s", file_path)
                self._notify(str(file_path), "failed", False, "File write timeout or invalid format.")
                return
        except OSError as exc:
            logger.error("Failed to access file stat during stabilization: %s", exc)
            self._notify(str(file_path), "failed", False, f"OS error: {exc}")
            return

        try:
            logger.info("Auto-indexing changed file: %s", file_path.name)
            db_id = self._use_case.index_changed_file(file_path)
            if db_id is None:
                logger.debug("Unchanged file skipped: %s", file_path.name)
                self._notify(str(file_path), "skipped", True, "")
                return

            logger.info("Successfully auto-indexed file. ID: %s", db_id)
            self._notify(str(file_path), "indexed", True, "")
        except Exception as exc:
            logger.error("Background indexing failed for file %s: %s", file_path, exc)
            self._notify(str(file_path), "failed", False, str(exc))

    def _process_deleted(self, file_path_str: str) -> None:
        file_path = Path(file_path_str).resolve()
        if not self._is_supported(file_path):
            return

        logger.info("Detected deleted image: %s", file_path.name)
        try:
            removed = self._use_case.remove_indexed_file(file_path)
            if removed:
                self._notify(str(file_path), "removed", True, "")
            else:
                self._notify(str(file_path), "skipped", True, "")
        except Exception as exc:
            logger.error("Failed to remove deleted file %s from index: %s", file_path, exc)
            self._notify(str(file_path), "failed", False, str(exc))

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule_process(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule_process(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        if getattr(event, "src_path", None):
            self._process_deleted(event.src_path)
        if getattr(event, "dest_path", None):
            self._schedule_process(event.dest_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._process_deleted(event.src_path)


class FolderMonitorController:
    """Manages starting and stopping Watchdog observers across multiple paths."""

    def __init__(
        self,
        indexing_use_case: IndexImagesUseCase,
        on_file_indexed_callback: Optional[AutoIndexCallback] = None,
    ) -> None:
        self._use_case = indexing_use_case
        self._on_indexed = on_file_indexed_callback
        self._observer = None
        self._active_watches: list = []
        self._handler: Optional[TileImageEventHandler] = None

    @property
    def is_running(self) -> bool:
        return self._observer is not None and self._observer.is_alive()

    def start_monitoring(self, folders: List[str]) -> None:
        if Observer is None or not WATCHDOG_AVAILABLE:
            logger.critical("watchdog package not installed! Cannot monitor folders.")
            raise ImportError(
                "watchdog package is required for folder monitoring. "
                "Install it with: pip install watchdog"
            )

        if self._observer is not None:
            self.stop_monitoring()

        self._observer = Observer()
        self._active_watches = []
        self._handler = TileImageEventHandler(
            indexing_use_case=self._use_case,
            on_file_indexed_callback=self._on_indexed,
        )

        for folder_str in folders:
            folder_path = Path(folder_str).resolve()
            if not folder_path.exists() or not folder_path.is_dir():
                logger.warning(
                    "Monitored folder path does not exist or is not a directory: %s",
                    folder_path,
                )
                continue

            logger.info("Registering watcher for directory: %s", folder_path)
            try:
                watch = self._observer.schedule(
                    self._handler,
                    path=str(folder_path),
                    recursive=True,
                )
                self._active_watches.append(watch)
            except Exception as exc:
                logger.error("Failed to start watchdog watcher for path %s: %s", folder_path, exc)

        if self._active_watches:
            self._observer.start()
            logger.info("Watchdog folder observer thread started successfully.")
        else:
            logger.warning("No valid folders monitored. Watchdog was not started.")

    def restart_monitoring(self, folders: List[str]) -> None:
        """Apply an updated watch-folder list without restarting the app."""
        if folders:
            self.start_monitoring(folders)
        else:
            self.stop_monitoring()

    def stop_monitoring(self) -> None:
        if self._observer is not None:
            logger.info("Stopping folder monitor observer thread...")
            self._observer.stop()
            self._observer.join()
            self._observer = None
            self._active_watches = []
            self._handler = None
            logger.info("Folder monitor observer stopped.")
