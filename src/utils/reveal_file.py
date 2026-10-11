"""Reveal a file in the OS file manager (select when the platform supports it)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

from src.utils.platform_info import is_linux, is_macos, is_windows

logger = logging.getLogger("tilevision.utils.reveal_file")


def reveal_in_file_manager(path: str | Path) -> tuple[bool, str]:
    """
    Reveal ``path`` in the platform file manager.

    Windows: Explorer with the file selected (``/select,``).
    macOS: Finder with the file selected (``open -R``).
    Linux: open the containing folder (no universal select API).

    Returns:
        ``(True, "")`` on success, or ``(False, error_message)`` when the
        path is missing / the launch fails.
    """
    target = Path(path)
    if not target.exists():
        return False, f"This file no longer exists:\n{target}"

    try:
        if is_windows():
            # explorer.exe often returns a non-zero exit code even on success —
            # never use check=True / subprocess.run here.
            subprocess.Popen(["explorer", "/select,", str(target)])
            return True, ""
        if is_macos():
            subprocess.run(["open", "-R", str(target)], check=False)
            return True, ""
        # Linux / other: open the parent folder.
        folder = target if target.is_dir() else target.parent
        if not folder.exists():
            return False, f"This folder no longer exists:\n{folder}"
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        return True, ""
    except Exception as exc:
        logger.error("Failed to reveal file %s: %s", target, exc)
        return False, f"Could not open file location:\n{exc}"


def reveal_action_label() -> str:
    """Platform-appropriate context-menu label for reveal_in_file_manager."""
    if is_windows():
        return "📍  Show in File Explorer"
    if is_macos():
        return "📍  Show in Finder"
    if is_linux():
        return "📍  Show in File Manager"
    return "📍  Show in File Manager"
