"""Tests for platform reveal-in-file-manager helper."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.reveal_file import reveal_action_label, reveal_in_file_manager


def test_reveal_missing_file(tmp_path):
    ok, err = reveal_in_file_manager(tmp_path / "missing.jpg")
    assert ok is False
    assert "no longer exists" in err


def test_reveal_windows_uses_explorer_select(tmp_path, monkeypatch):
    target = tmp_path / "tile.jpg"
    target.write_bytes(b"x")
    monkeypatch.setattr("src.utils.reveal_file.is_windows", lambda: True)
    monkeypatch.setattr("src.utils.reveal_file.is_macos", lambda: False)
    monkeypatch.setattr("src.utils.reveal_file.is_linux", lambda: False)
    with patch("src.utils.reveal_file.subprocess.Popen") as popen:
        ok, err = reveal_in_file_manager(target)
    assert ok is True
    assert err == ""
    popen.assert_called_once()
    args = popen.call_args[0][0]
    assert args[0] == "explorer"
    assert args[1] == "/select,"
    assert args[2] == str(target)


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
