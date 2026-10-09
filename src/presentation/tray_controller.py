"""
System tray icon for TileVision AI background (minimize-to-tray) mode.

When ``settings.minimize_to_tray_on_close`` is enabled, closing the main
window hides it and keeps the process (and folder monitor) alive via a
``QSystemTrayIcon``. Full exit is only via the tray "Quit" action.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QMenu, QSystemTrayIcon, QWidget

from src.utils.platform_info import app_icon_path

logger = logging.getLogger("tilevision.presentation.tray_controller")

TRAY_TOOLTIP = "TileVision AI is running in the background"


class TrayController(QObject):
    """Owns the system tray icon and Open / Quit menu actions."""

    open_requested = Signal()
    quit_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tray: Optional[QSystemTrayIcon] = None
        self._menu: Optional[QMenu] = None

    @property
    def is_visible(self) -> bool:
        return self._tray is not None and self._tray.isVisible()

    @staticmethod
    def is_available() -> bool:
        return bool(QSystemTrayIcon.isSystemTrayAvailable())

    def ensure_shown(self) -> bool:
        """Create and show the tray icon. Returns False if the OS has no tray."""
        if not self.is_available():
            logger.warning(
                "System tray is unavailable — cannot keep TileVision AI in the background."
            )
            return False

        if self._tray is None:
            icon = QIcon()
            path = app_icon_path()
            if path is not None:
                icon = QIcon(str(path))
            self._tray = QSystemTrayIcon(icon, self)
            self._tray.setToolTip(TRAY_TOOLTIP)

            self._menu = QMenu()
            open_action = QAction("Open", self._menu)
            open_action.triggered.connect(self.open_requested.emit)
            quit_action = QAction("Quit", self._menu)
            quit_action.triggered.connect(self.quit_requested.emit)
            self._menu.addAction(open_action)
            self._menu.addSeparator()
            self._menu.addAction(quit_action)
            self._tray.setContextMenu(self._menu)
            self._tray.activated.connect(self._on_activated)

        self._tray.show()
        logger.info("System tray icon shown.")
        return True

    def hide(self) -> None:
        """Hide the tray icon (e.g. on license cutover or setting turned off)."""
        if self._tray is not None:
            self._tray.hide()
            logger.info("System tray icon hidden.")

    def destroy(self) -> None:
        """Remove the tray icon entirely."""
        if self._tray is not None:
            self._tray.hide()
            self._tray.setContextMenu(None)
            self._tray.deleteLater()
            self._tray = None
        if self._menu is not None:
            self._menu.deleteLater()
            self._menu = None

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.open_requested.emit()
