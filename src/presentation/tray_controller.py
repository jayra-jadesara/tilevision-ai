"""
System tray icon for TileVision AI background (minimize-to-tray) mode.

When ``settings.minimize_to_tray_on_close`` is enabled, closing the main
window hides the UI and keeps the process (and folder monitor) alive via a
``QSystemTrayIcon``. Full exit is only via the tray "Quit" action.

Under headless Qt platforms (``offscreen`` / ``minimal`` / ``null``),
``QSystemTrayIcon`` must not be imported or probed — on some Linux/macOS CI
images that initializes a broken tray backend and SIGSEGVs at interpreter
shutdown after an otherwise green pytest suite.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QWidget

from src.utils.platform_info import app_icon_path

logger = logging.getLogger("tilevision.presentation.tray_controller")

TRAY_TOOLTIP = "TileVision AI is running in the background"

# QPA platforms where a real system tray must not be constructed or probed.
_HEADLESS_QT_PLATFORMS = frozenset({"offscreen", "minimal", "null", "vnc"})


class TrayController(QObject):
    """Owns the system tray icon and Open / Quit menu actions."""

    open_requested = Signal()
    quit_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tray: Any = None
        self._menu: Any = None
        self._open_action: Any = None
        self._quit_action: Any = None

    @property
    def is_visible(self) -> bool:
        return self._tray is not None and self._tray.isVisible()

    @staticmethod
    def is_headless_platform() -> bool:
        """True when the active Qt platform cannot host a real system tray."""
        app = QApplication.instance()
        if app is None:
            return True
        name = str(app.platformName() or "").strip().lower()
        return name in _HEADLESS_QT_PLATFORMS

    @staticmethod
    def is_available() -> bool:
        """
        True only when a real desktop tray can be used safely.

        On headless QPA platforms this returns False without importing or
        calling into ``QSystemTrayIcon`` (that probe alone can crash CI).
        """
        if TrayController.is_headless_platform():
            return False
        from PySide6.QtWidgets import QSystemTrayIcon

        return bool(QSystemTrayIcon.isSystemTrayAvailable())

    def ensure_shown(self) -> bool:
        """Create and show the tray icon. Returns False if the OS has no tray."""
        if not self.is_available():
            logger.warning(
                "System tray is unavailable (platform=%r) — cannot keep "
                "TileVision AI in the background.",
                QApplication.instance().platformName()
                if QApplication.instance() is not None
                else None,
            )
            return False

        # Lazy imports: keep headless pytest processes free of QSystemTrayIcon.
        from PySide6.QtGui import QAction, QIcon
        from PySide6.QtWidgets import QMenu, QSystemTrayIcon

        if self._tray is None:
            icon = QIcon()
            path = app_icon_path()
            if path is not None:
                icon = QIcon(str(path))
            self._tray = QSystemTrayIcon(icon, self)
            self._tray.setToolTip(TRAY_TOOLTIP)

            self._menu = QMenu()
            self._open_action = QAction("Open", self._menu)
            self._open_action.triggered.connect(self.open_requested.emit)
            self._quit_action = QAction("Quit", self._menu)
            self._quit_action.triggered.connect(self.quit_requested.emit)
            self._menu.addAction(self._open_action)
            self._menu.addSeparator()
            self._menu.addAction(self._quit_action)
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
        """
        Synchronously tear down native tray objects.

        Order: disconnect → clear menu actions → drop context menu → delete
        menu → delete tray. Uses ``shiboken6.delete`` instead of
        ``deleteLater()`` so cleanup does not race interpreter shutdown.
        """
        import shiboken6

        tray = self._tray
        menu = self._menu
        open_action = self._open_action
        quit_action = self._quit_action
        self._tray = None
        self._menu = None
        self._open_action = None
        self._quit_action = None

        if tray is not None and shiboken6.isValid(tray):
            tray.hide()
            try:
                tray.activated.disconnect(self._on_activated)
            except (RuntimeError, TypeError):
                pass
            tray.setContextMenu(None)

        for action in (open_action, quit_action):
            if action is None or not shiboken6.isValid(action):
                continue
            try:
                action.triggered.disconnect()
            except (RuntimeError, TypeError):
                pass
            if menu is not None and shiboken6.isValid(menu):
                menu.removeAction(action)
            action.setParent(None)
            shiboken6.delete(action)

        if menu is not None and shiboken6.isValid(menu):
            menu.clear()
            menu.setParent(None)
            shiboken6.delete(menu)

        if tray is not None and shiboken6.isValid(tray):
            tray.setParent(None)
            shiboken6.delete(tray)

        app = QApplication.instance()
        if app is not None:
            app.processEvents()

        logger.info("System tray icon destroyed.")

    def _on_activated(self, reason: Any) -> None:
        from PySide6.QtWidgets import QSystemTrayIcon

        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.open_requested.emit()
