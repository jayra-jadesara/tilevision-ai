"""Brand and navigation icon helpers for TileVision AI."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap

_RESOURCES_DIR = Path(__file__).resolve().parents[1] / "resources"
_ICONS_DIR = _RESOURCES_DIR / "icons"

APP_ICON_PATH = _RESOURCES_DIR / "app_icon.ico"
APP_ICON_PNG_PATH = _RESOURCES_DIR / "app_icon.png"
LOGO_SMALL_PATH = _RESOURCES_DIR / "logo_small.png"


def nav_icon(icon_name: str, theme: str) -> QIcon:
    """Return a theme-aware sidebar navigation icon."""
    safe_theme = "light" if theme == "light" else "dark"
    icon_path = _ICONS_DIR / safe_theme / f"nav_{icon_name}.svg"
    if icon_path.exists():
        return QIcon(str(icon_path))
    return QIcon()


def logo_pixmap(size: int = 56) -> QPixmap:
    """Return the sidebar logo scaled to the requested size."""
    pixmap = QPixmap(str(LOGO_SMALL_PATH))
    if pixmap.isNull():
        return QPixmap()
    return pixmap.scaled(
        size,
        size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def nav_icon_size() -> QSize:
    return QSize(24, 24)


def license_status_icon(theme: str, *, is_trial: bool = False) -> QIcon:
    """Return a theme-aware license badge icon for the status bar."""
    icon_name = "license_trial" if is_trial else "license"
    safe_theme = "light" if theme == "light" else "dark"
    icon_path = _ICONS_DIR / safe_theme / f"nav_{icon_name}.svg"
    if icon_path.exists():
        return QIcon(str(icon_path))
    return QIcon()


def license_status_icon_size() -> QSize:
    return QSize(16, 16)
