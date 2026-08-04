"""Tests for in-app update installer (Windows silent + Mac DMG apply)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtWidgets import QApplication

from src.utils import update_installer as ui


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_windows_silent_install_args_include_close_and_silent(tmp_path):
    setup = tmp_path / "TileVisionAI-Setup-1.2.5.exe"
    setup.write_bytes(b"MZ")
    args = ui.windows_silent_install_args(setup)
    assert args[0] == str(setup)
    assert "/VERYSILENT" in args
    assert "/SUPPRESSMSGBOXES" in args
    assert "/CLOSEAPPLICATIONS" in args
    assert "/FORCECLOSEAPPLICATIONS" in args
    assert "/NORESTART" in args


def test_launch_windows_silent_installer_rejects_non_exe(tmp_path):
    bad = tmp_path / "update.dmg"
    bad.write_bytes(b"x")
    with pytest.raises(ui.UpdateInstallError, match="\\.exe"):
        ui.launch_windows_silent_installer(bad)


def test_launch_windows_silent_installer_spawns_detached(tmp_path, monkeypatch):
    setup = tmp_path / "TileVisionAI-Setup-1.2.5.exe"
    setup.write_bytes(b"MZ")
    fake = MagicMock()
    monkeypatch.setattr(ui.subprocess, "Popen", fake)
    monkeypatch.setattr(ui.sys, "platform", "win32")

    ui.launch_windows_silent_installer(setup)

    fake.assert_called_once()
    args, kwargs = fake.call_args
    assert args[0][0] == "cmd.exe"
    assert args[0][1] == "/c"
    script = Path(args[0][2])
    assert script.suffix.lower() == ".cmd"
    body = script.read_text(encoding="utf-8")
    assert "/VERYSILENT" in body
    assert "TileVision AI" in body
    assert kwargs.get("start_new_session") is True


def test_build_windows_apply_script_waits_and_relaunches(tmp_path):
    setup = tmp_path / "TileVisionAI-Setup-1.2.6.exe"
    setup.write_bytes(b"MZ")
    script = ui.build_windows_apply_script(setup, wait_pid=9999)
    assert "WAIT_PID=9999" in script
    assert "/VERYSILENT" in script
    assert "/SUPPRESSMSGBOXES" in script
    assert "FORCECLOSEAPPLICATIONS" in script
    assert "TileVisionAI.exe" in script


def test_build_macos_apply_script_waits_and_replaces(tmp_path):
    dmg = tmp_path / "TileVisionAI-macOS-Intel-1.2.5.dmg"
    dmg.write_bytes(b"dmg")
    script = ui.build_macos_apply_script(dmg, wait_pid=4242, dest_app=Path("/Applications/TileVisionAI.app"))
    assert "WAIT_PID=4242" in script
    assert "hdiutil attach" in script
    assert "ditto" in script
    assert "open -n" in script
    assert "TileVisionAI.app" in script
    assert str(dmg.resolve()) in script or "TileVisionAI-macOS-Intel-1.2.5.dmg" in script


def test_write_macos_apply_script_is_executable(tmp_path):
    dmg = tmp_path / "app.dmg"
    dmg.write_bytes(b"dmg")
    path = ui.write_macos_apply_script(dmg, wait_pid=1, script_path=tmp_path / "apply.sh")
    assert path.exists()
    # Unix execute bits are meaningful on Darwin/Linux; Windows NTFS often
    # reports no +x even after chmod — content is what matters for the helper.
    body = path.read_text(encoding="utf-8")
    assert body.startswith("#!/bin/bash")
    assert "hdiutil attach" in body
    if sys.platform != "win32":
        assert path.stat().st_mode & 0o111


def test_launch_macos_dmg_installer_rejects_non_dmg(tmp_path):
    bad = tmp_path / "setup.exe"
    bad.write_bytes(b"MZ")
    with pytest.raises(ui.UpdateInstallError, match="\\.dmg"):
        ui.launch_macos_dmg_installer(bad)


def test_launch_update_installer_dispatches_windows(tmp_path, monkeypatch):
    setup = tmp_path / "setup.exe"
    setup.write_bytes(b"MZ")
    called = {}

    def _win(path):
        called["path"] = path
        return MagicMock()

    monkeypatch.setattr(ui.sys, "platform", "win32")
    monkeypatch.setattr(ui, "launch_windows_silent_installer", _win)
    ui.launch_update_installer(setup)
    assert called["path"] == setup


def test_launch_update_installer_dispatches_mac(tmp_path, monkeypatch):
    dmg = tmp_path / "app.dmg"
    dmg.write_bytes(b"dmg")
    called = {}

    def _mac(path, dest_app=None):
        called["path"] = path
        return MagicMock()

    monkeypatch.setattr(ui.sys, "platform", "darwin")
    monkeypatch.setattr(ui, "launch_macos_dmg_installer", _mac)
    ui.launch_update_installer(dmg)
    assert called["path"] == dmg


def test_force_quit_for_update_flag():
    ui.reset_force_quit_for_update_for_tests()
    assert ui.is_force_quit_for_update() is False
    ui.begin_force_quit_for_update()
    assert ui.is_force_quit_for_update() is True
    ui.reset_force_quit_for_update_for_tests()
    assert ui.is_force_quit_for_update() is False


def test_update_dialog_reuses_cached_installer(qapp, tmp_path, monkeypatch):
    from src.config.settings import AppSettings
    from src.presentation.views.update_dialog import UpdateAvailableDialog
    from src.utils.update_check import UpdateInfo

    installer = tmp_path / "TileVisionAI-Setup-9.9.9.exe"
    installer.write_bytes(b"MZ" + b"0" * 100)

    settings = AppSettings(config_dir=tmp_path / "cfg")
    settings.set_pending_update("9.9.9", str(installer))

    info = UpdateInfo(
        current_version="1.0.0",
        latest_version="9.9.9",
        release_notes="test",
        download_url="https://example.com/TileVisionAI-Setup-9.9.9.exe",
    )
    launched = {}

    monkeypatch.setattr(
        "src.presentation.views.update_dialog.resolve_cached_installer",
        lambda url, preferred_path=None: installer,
    )
    monkeypatch.setattr(
        "src.presentation.views.update_dialog.launch_update_installer",
        lambda path: launched.setdefault("path", path),
    )
    monkeypatch.setattr(
        "src.presentation.views.update_dialog.begin_force_quit_for_update",
        lambda: None,
    )

    class _App:
        def quit(self):
            launched["quit"] = True

    monkeypatch.setattr(
        "src.presentation.views.update_dialog.QApplication.instance",
        lambda: _App(),
    )

    dialog = UpdateAvailableDialog(
        info,
        theme="light",
        auto_start_download=True,
        auto_install_after_download=True,
        settings=settings,
    )
    from PySide6.QtCore import QCoreApplication

    QCoreApplication.processEvents()
    dialog._start_download_or_reuse_cache()
    assert dialog._downloaded_path == installer
    assert settings.pending_update_installer_path == str(installer)
    dialog._on_install_and_restart()
    assert launched["path"] == installer
    dialog.close()


def test_update_dialog_cancels_indexing_before_quit(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QWidget

    from src.presentation.views.update_dialog import UpdateAvailableDialog
    from src.utils.update_check import UpdateInfo

    installer = tmp_path / "setup.exe"
    installer.write_bytes(b"MZ")
    info = UpdateInfo(
        current_version="1.0.0",
        latest_version="9.9.9",
        release_notes="test",
        download_url="https://example.com/setup.exe",
    )

    class FakeVM:
        def __init__(self):
            self.cancelled = False

        def cancel_indexing(self):
            self.cancelled = True

    parent = QWidget()
    parent._indexing_viewmodel = FakeVM()  # type: ignore[attr-defined]
    force_calls = {"n": 0}

    monkeypatch.setattr(
        "src.presentation.views.update_dialog.launch_update_installer",
        lambda path: None,
    )
    monkeypatch.setattr(
        "src.presentation.views.update_dialog.begin_force_quit_for_update",
        lambda: force_calls.__setitem__("n", force_calls["n"] + 1),
    )
    monkeypatch.setattr(
        "src.presentation.views.update_dialog.QApplication.instance",
        lambda: type("A", (), {"quit": staticmethod(lambda: None)})(),
    )

    dialog = UpdateAvailableDialog(
        info,
        theme="light",
        parent=parent,
        auto_start_download=False,
        auto_install_after_download=False,
    )
    dialog._downloaded_path = installer
    dialog._on_install_and_restart()
    assert parent._indexing_viewmodel.cancelled is True  # type: ignore[attr-defined]
    assert force_calls["n"] >= 1
    dialog.close()
    parent.close()


def test_update_dialog_auto_installs_after_download(qapp, tmp_path, monkeypatch):
    from src.presentation.views.update_dialog import UpdateAvailableDialog
    from src.utils.update_check import UpdateInfo

    installer = tmp_path / "TileVisionAI-Setup-9.9.9.exe"
    installer.write_bytes(b"MZ")
    info = UpdateInfo(
        current_version="1.0.0",
        latest_version="9.9.9",
        release_notes="test",
        download_url="https://example.com/setup.exe",
    )
    launched = {}
    quit_calls = {"n": 0}

    monkeypatch.setattr(
        "src.presentation.views.update_dialog.launch_update_installer",
        lambda path: launched.setdefault("path", path),
    )
    monkeypatch.setattr(
        "src.presentation.views.update_dialog.begin_force_quit_for_update",
        lambda: None,
    )

    class _App:
        def quit(self):
            quit_calls["n"] += 1

    monkeypatch.setattr(
        "src.presentation.views.update_dialog.QApplication.instance",
        lambda: _App(),
    )

    dialog = UpdateAvailableDialog(
        info,
        theme="light",
        auto_start_download=False,
        auto_install_after_download=True,
    )
    dialog._on_download_ok(str(installer))
    # Process the queued install timer.
    from PySide6.QtCore import QCoreApplication

    QCoreApplication.processEvents()
    dialog._on_install_and_restart()
    assert launched["path"] == installer
    hard = {"n": 0}
    monkeypatch.setattr(
        "src.presentation.views.update_dialog.os._exit",
        lambda code: hard.__setitem__("n", hard["n"] + 1),
    )
    dialog._quit_for_install()
    assert quit_calls["n"] == 1
    # Fire the hard-exit timer.
    QCoreApplication.processEvents()
    dialog._hard_exit_for_install()
    assert hard["n"] == 1
    dialog.close()
