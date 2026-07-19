"""Background worker for online update checks."""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QThread, Signal

from src.utils.update_check import UpdateInfo, check_for_updates

logger = logging.getLogger("tilevision.presentation.workers.update_check")


class UpdateCheckWorker(QThread):
    """Fetch the update manifest without blocking the UI thread."""

    finished_check = Signal(object, object)  # UpdateInfo | None, error str | None

    def __init__(
        self,
        *,
        manifest_url: str,
        current_version: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._manifest_url = manifest_url
        self._current_version = current_version

    def run(self) -> None:
        try:
            info = check_for_updates(
                current_version=self._current_version,
                manifest_url=self._manifest_url,
            )
            self.finished_check.emit(info, None)
        except Exception as exc:
            logger.debug("Update check failed: %s", exc)
            self.finished_check.emit(None, str(exc))
