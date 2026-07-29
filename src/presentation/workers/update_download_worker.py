"""Background worker that downloads an update installer with progress."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

from src.utils.update_downloader import (
    DEFAULT_CONNECTIONS,
    DownloadCancelled,
    download_update_file,
)

logger = logging.getLogger("tilevision.presentation.workers.update_download_worker")


class UpdateDownloadWorker(QThread):
    """Download a release installer on a background thread."""

    progress = Signal(int, int, float)  # received, total, bytes_per_sec
    finished_ok = Signal(str)  # absolute path
    finished_error = Signal(str)
    finished_cancelled = Signal()

    def __init__(
        self,
        url: str,
        *,
        dest_dir: Optional[Path] = None,
        connections: int = DEFAULT_CONNECTIONS,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._url = url
        self._dest_dir = dest_dir
        self._connections = connections
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            path = download_update_file(
                self._url,
                self._dest_dir,
                connections=self._connections,
                cancel_event=self._cancel,
                progress=self._emit_progress,
            )
            if self._cancel.is_set():
                self.finished_cancelled.emit()
                return
            self.finished_ok.emit(str(path))
        except DownloadCancelled:
            logger.info("Update download cancelled by user.")
            self.finished_cancelled.emit()
        except Exception as exc:
            logger.error("Update download failed: %s", exc)
            self.finished_error.emit(str(exc))

    def _emit_progress(self, received: int, total: int, speed: float) -> None:
        self.progress.emit(int(received), int(total), float(speed))
