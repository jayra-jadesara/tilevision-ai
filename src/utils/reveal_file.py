"""Reveal a file in the OS file manager (select when the platform supports it)."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

from src.utils.platform_info import is_linux, is_macos, is_windows

logger = logging.getLogger("tilevision.utils.reveal_file")


def _path_missing(target: Path) -> bool:
    """Fast existence check — avoid slow pathlib edge cases on network drives."""
    try:
        return not os.path.exists(os.fspath(target))
    except OSError:
        return True


def _reveal_windows(target: Path) -> None:
    """
    Open Explorer with the file selected, without blocking the UI thread.

    Prefer Shell32 ``SHOpenFolderAndSelectItems`` (snappy when Explorer is
    already running); fall back to a detached ``explorer /select,`` launch.
    """
    path_str = os.path.normpath(str(target))
    try:
        import ctypes
        from ctypes import wintypes

        shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
        ole32 = ctypes.windll.ole32  # type: ignore[attr-defined]

        ole32.CoInitialize(None)
        try:
            # ILCreateFromPathW returns a PIDL that must be freed with ILFree.
            ILCreateFromPathW = shell32.ILCreateFromPathW
            ILCreateFromPathW.argtypes = [wintypes.LPCWSTR]
            ILCreateFromPathW.restype = ctypes.c_void_p

            ILFree = shell32.ILFree
            ILFree.argtypes = [ctypes.c_void_p]

            SHOpenFolderAndSelectItems = shell32.SHOpenFolderAndSelectItems
            SHOpenFolderAndSelectItems.argtypes = [
                ctypes.c_void_p,
                wintypes.UINT,
                ctypes.c_void_p,
                wintypes.DWORD,
            ]
            SHOpenFolderAndSelectItems.restype = ctypes.HRESULT

            pidl = ILCreateFromPathW(path_str)
            if pidl:
                try:
                    hr = SHOpenFolderAndSelectItems(pidl, 0, None, 0)
                    if hr >= 0:
                        return
                finally:
                    ILFree(pidl)
        finally:
            ole32.CoUninitialize()
    except Exception as exc:
        logger.debug("SHOpenFolderAndSelectItems unavailable, falling back: %s", exc)

    # Detached explorer launch — never wait on the process (explorer often
    # returns non-zero even on success and can appear to hang if joined).
    creationflags = 0
    if hasattr(subprocess, "DETACHED_PROCESS"):
        creationflags |= subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags |= subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    subprocess.Popen(
        ["explorer", "/select,", path_str],
        close_fds=True,
        creationflags=creationflags,
    )


def reveal_in_file_manager(path: str | Path) -> tuple[bool, str]:
    """
    Reveal ``path`` in the platform file manager.

    Windows: Explorer with the file selected (Shell32, else ``/select,``).
    macOS: Finder with the file selected (``open -R``, non-blocking).
    Linux: open the containing folder (no universal select API).

    Returns:
        ``(True, "")`` on success, or ``(False, error_message)`` when the
        path is missing / the launch fails.
    """
    target = Path(path)
    if _path_missing(target):
        return False, f"This file no longer exists:\n{target}"

    try:
        if is_windows():
            _reveal_windows(target)
            return True, ""
        if is_macos():
            # Popen — do not block the UI waiting for `open` to return.
            subprocess.Popen(
                ["open", "-R", str(target)],
                close_fds=True,
            )
            return True, ""
        # Linux / other: open the parent folder.
        folder = target if target.is_dir() else target.parent
        if _path_missing(folder):
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
