"""Tests for platform reveal-in-file-manager helper."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.reveal_file import (
    _reveal_windows,
    reveal_action_label,
    reveal_in_file_manager,
)


def test_reveal_missing_file(tmp_path):
    ok, err = reveal_in_file_manager(tmp_path / "missing.jpg")
    assert ok is False
    assert "no longer exists" in err


def test_reveal_windows_delegates_to_helper(tmp_path, monkeypatch):
    target = tmp_path / "tile.jpg"
    target.write_bytes(b"x")
    monkeypatch.setattr("src.utils.reveal_file.is_windows", lambda: True)
    monkeypatch.setattr("src.utils.reveal_file.is_macos", lambda: False)
    monkeypatch.setattr("src.utils.reveal_file.is_linux", lambda: False)
    with patch("src.utils.reveal_file._reveal_windows") as helper:
        ok, err = reveal_in_file_manager(target)
    assert ok is True
    assert err == ""
    helper.assert_called_once()
    assert helper.call_args[0][0] == target


def test_reveal_windows_shell32_success_skips_explorer_popen(tmp_path, monkeypatch):
    """When Shell32 select succeeds, do not spawn explorer.exe."""
    target = tmp_path / "tile.jpg"
    target.write_bytes(b"x")

    fake_shell32 = MagicMock()
    fake_ole32 = MagicMock()
    fake_shell32.ILCreateFromPathW.return_value = 0xABC  # non-null PIDL
    fake_shell32.SHOpenFolderAndSelectItems.return_value = 0  # S_OK

    fake_ctypes = MagicMock()
    fake_ctypes.windll.shell32 = fake_shell32
    fake_ctypes.windll.ole32 = fake_ole32
    fake_ctypes.c_void_p = MagicMock()
    fake_ctypes.HRESULT = MagicMock()
    fake_ctypes.wintypes = MagicMock()

    with patch.dict(sys.modules, {"ctypes": fake_ctypes, "ctypes.wintypes": fake_ctypes.wintypes}):
        with patch("src.utils.reveal_file.subprocess.Popen") as popen:
            # Re-import path is internal; call helper directly with patched modules.
            # _reveal_windows does `import ctypes` locally — sys.modules patch applies.
            _reveal_windows(target)

    popen.assert_not_called()
    fake_shell32.SHOpenFolderAndSelectItems.assert_called_once()
    fake_shell32.ILFree.assert_called_once_with(0xABC)


def test_reveal_windows_falls_back_to_explorer_select(tmp_path, monkeypatch):
    """If Shell32 is unavailable, fall back to detached explorer /select,."""
    target = tmp_path / "tile.jpg"
    target.write_bytes(b"x")

    # Force the Shell32 branch to raise so we hit the Popen fallback.
    boom = MagicMock()
    boom.windll.shell32 = property(lambda self: (_ for _ in ()).throw(AttributeError("no shell32")))
    # Simpler: make windll access raise
    broken_ctypes = MagicMock()
    type(broken_ctypes).windll = property(
        lambda self: (_ for _ in ()).throw(AttributeError("no windll"))
    )

    with patch.dict(sys.modules, {"ctypes": broken_ctypes}):
        with patch("src.utils.reveal_file.subprocess.Popen") as popen:
            _reveal_windows(target)

    popen.assert_called_once()
    args = popen.call_args[0][0]
    assert args[0] == "explorer"
    assert args[1] == "/select,"
    # Windows normalizes separators; compare via Path.
    assert Path(args[2]) == target


def test_reveal_macos_uses_open_r(tmp_path, monkeypatch):
    target = tmp_path / "tile.jpg"
    target.write_bytes(b"x")
    monkeypatch.setattr("src.utils.reveal_file.is_windows", lambda: False)
    monkeypatch.setattr("src.utils.reveal_file.is_macos", lambda: True)
    monkeypatch.setattr("src.utils.reveal_file.is_linux", lambda: False)
    with patch("src.utils.reveal_file.subprocess.Popen") as popen:
        ok, err = reveal_in_file_manager(target)
    assert ok is True
    popen.assert_called_once()
    assert popen.call_args[0][0][:2] == ["open", "-R"]


def test_reveal_linux_opens_parent_folder(tmp_path, monkeypatch):
    target = tmp_path / "tile.jpg"
    target.write_bytes(b"x")
    monkeypatch.setattr("src.utils.reveal_file.is_windows", lambda: False)
    monkeypatch.setattr("src.utils.reveal_file.is_macos", lambda: False)
    monkeypatch.setattr("src.utils.reveal_file.is_linux", lambda: True)
    with patch("src.utils.reveal_file.QDesktopServices.openUrl") as open_url:
        ok, err = reveal_in_file_manager(target)
    assert ok is True
    open_url.assert_called_once()


def test_reveal_action_label_by_platform(monkeypatch):
    monkeypatch.setattr("src.utils.reveal_file.is_windows", lambda: True)
    monkeypatch.setattr("src.utils.reveal_file.is_macos", lambda: False)
    monkeypatch.setattr("src.utils.reveal_file.is_linux", lambda: False)
    assert "Explorer" in reveal_action_label()

    monkeypatch.setattr("src.utils.reveal_file.is_windows", lambda: False)
    monkeypatch.setattr("src.utils.reveal_file.is_macos", lambda: True)
    assert "Finder" in reveal_action_label()
