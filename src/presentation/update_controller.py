"""Coordinates startup and manual update checks."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox, QWidget

from src.config.settings import AppSettings
from src.presentation.views.update_dialog import UpdateAvailableDialog
from src.presentation.workers.update_check_worker import UpdateCheckWorker
from src.utils.update_check import DEFAULT_MANIFEST_URL, UpdateInfo, check_for_updates
from src.version import APP_VERSION

logger = logging.getLogger("tilevision.update_controller")

_CHECK_INTERVAL_HOURS = 24
_STARTUP_DELAY_MS = 8000


class UpdateController:
    """Non-blocking update notifications for packaged customer builds."""

    def __init__(self, settings: AppSettings, *, theme: str = "light") -> None:
        self._settings = settings
        self._theme = theme
        self._worker: Optional[UpdateCheckWorker] = None
        self._parent: Optional[QWidget] = None

    def schedule_startup_check(self, parent: QWidget) -> None:
        import sys

        if not getattr(sys, "frozen", False):
            return
        if not self._settings.check_for_updates:
            return
        if not self._should_check_now():
            return

        self._parent = parent
        QTimer.singleShot(_STARTUP_DELAY_MS, lambda: self._start_check(silent=True))

    def check_now(self, parent: QWidget) -> None:
        self._parent = parent
        self._start_check(silent=False)

    def _should_check_now(self) -> bool:
        skipped = self._settings.skipped_update_version
        if skipped and skipped == self._settings.last_seen_update_version:
            return False

        last = self._settings.last_update_check_at
        if not last:
            return True
        try:
            previous = datetime.fromisoformat(last)
            if previous.tzinfo is None:
                previous = previous.replace(tzinfo=timezone.utc)
            elapsed_hours = (datetime.now(timezone.utc) - previous).total_seconds() / 3600
            return elapsed_hours >= _CHECK_INTERVAL_HOURS
        except ValueError:
            return True

    def _mark_checked(self) -> None:
        self._settings.last_update_check_at = datetime.now(timezone.utc).isoformat()

    def _start_check(self, *, silent: bool) -> None:
        if self._worker is not None and self._worker.isRunning():
            return

        manifest_url = self._settings.update_manifest_url or DEFAULT_MANIFEST_URL
        self._worker = UpdateCheckWorker(
            manifest_url=manifest_url,
            current_version=APP_VERSION,
            parent=self._parent,
        )
        self._worker.finished_check.connect(
            lambda info, error: self._on_finished(info, error, silent=silent)
        )
        self._worker.start()

    def _on_finished(self, info: object, error: Optional[str], *, silent: bool) -> None:
        self._mark_checked()
        parent = self._parent

        if error:
            if not silent and parent is not None:
                QMessageBox.information(
                    parent,
                    "No Update Found",
                    "Could not reach the update server.\n\n"
                    "Check your internet connection, or download the latest installer "
                    "from your TileVision vendor.",
                )
            return

        if info is None:
            if not silent and parent is not None:
                QMessageBox.information(
                    parent,
                    "Up to Date",
                    f"TileVision AI {APP_VERSION} is the latest version.",
                )
            return

        if not isinstance(info, UpdateInfo) or not info.is_newer:
            return

        if info.latest_version == self._settings.skipped_update_version:
            return

        self._settings.last_seen_update_version = info.latest_version
        if parent is None:
            return

        dialog = UpdateAvailableDialog(info, theme=self._theme, parent=parent)
        result = dialog.exec()
        if result == 2:
            self._settings.skipped_update_version = info.latest_version
