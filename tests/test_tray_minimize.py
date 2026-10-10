"""Tests for minimize-to-tray state transitions (headless/offscreen safe).

IMPORTANT: Never import or probe ``QSystemTrayIcon`` in this module's process
under ``QT_QPA_PLATFORM=offscreen``. On ubuntu-latest / macos-15 that can
initialize a broken tray backend and SIGSEGV at interpreter shutdown after
an otherwise green suite. Forced native construct/destroy runs only in an
isolated child process.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

from src.config.settings import AppSettings
from src.presentation.tray_controller import TrayController, TRAY_TOOLTIP
from src.presentation.viewmodels.indexing_viewmodel import IndexingViewModel
from src.presentation.views.main_window import MainWindow


class _FakeIndexUseCase:
    pass


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    return app


def _make_main_window(
    tmp_path: Path,
    catalogue_master_service,
    *,
    minimize: bool = False,
) -> MainWindow:
    settings = AppSettings(config_dir=tmp_path / "cfg")
    settings.minimize_to_tray_on_close = minimize
    indexing_vm = IndexingViewModel(use_case=_FakeIndexUseCase())
    return MainWindow(
        indexing_viewmodel=indexing_vm,
        settings=settings,
        catalogue_master_service=catalogue_master_service,
    )


def test_minimize_to_tray_setting_default_off(tmp_path: Path) -> None:
    settings = AppSettings(config_dir=tmp_path / "cfg")
    assert settings.minimize_to_tray_on_close is False


def test_minimize_to_tray_setting_persists(tmp_path: Path) -> None:
    settings = AppSettings(config_dir=tmp_path / "cfg")
    settings.minimize_to_tray_on_close = True
    reloaded = AppSettings(config_dir=tmp_path / "cfg")
    assert reloaded.minimize_to_tray_on_close is True


def test_close_event_hides_when_tray_enabled(
    qapp, tmp_path: Path, catalogue_master_service
) -> None:
    window = _make_main_window(tmp_path, catalogue_master_service, minimize=True)
    window.show()
    assert window.isVisible()

    event = QCloseEvent()
    window.closeEvent(event)

    assert event.isAccepted() is False
    assert window.isVisible() is False
    window.deleteLater()
    qapp.processEvents()


def test_close_event_accepts_when_tray_disabled(
    qapp, tmp_path: Path, catalogue_master_service
) -> None:
    window = _make_main_window(tmp_path, catalogue_master_service, minimize=False)
    window.show()

    event = QCloseEvent()
    window.closeEvent(event)

    assert event.isAccepted() is True
    window.deleteLater()
    qapp.processEvents()


def test_request_quit_bypasses_tray_minimize(
    qapp, tmp_path: Path, catalogue_master_service
) -> None:
    window = _make_main_window(tmp_path, catalogue_master_service, minimize=True)
    window.show()

    window._force_quit_requested = True
    event = QCloseEvent()
    window.closeEvent(event)

    assert event.isAccepted() is True
    window.deleteLater()
    qapp.processEvents()


def test_restore_from_tray(qapp, tmp_path: Path, catalogue_master_service) -> None:
    window = _make_main_window(tmp_path, catalogue_master_service, minimize=True)
    window.show()
    window.hide()
    assert not window.isVisible()
    window.restore_from_tray()
    assert window.isVisible()
    window.deleteLater()
    qapp.processEvents()


def test_tray_reports_ci_platform_diagnostics(qapp) -> None:
    """Task 1 evidence: log platform classification under CI's QPA."""
    platform = qapp.platformName()
    headless = TrayController.is_headless_platform()
    controller_available = TrayController.is_available()
    print(
        f"TRAY_DIAG platformName={platform!r} "
        f"isSystemTrayAvailable=not_probed_in_main_process "
        f"TrayController.is_available={controller_available} "
        f"is_headless_platform={headless}",
        flush=True,
    )
    if str(platform).lower() in {"offscreen", "minimal", "null", "vnc"}:
        assert headless is True
        assert controller_available is False


def test_offscreen_never_constructs_tray_even_if_qt_claims_available(
    qapp, monkeypatch
) -> None:
    """
    Guard against offscreen lying via isSystemTrayAvailable().

    Does not import or monkeypatch QSystemTrayIcon in this process.
    """
    monkeypatch.setattr(
        TrayController, "is_headless_platform", staticmethod(lambda: True)
    )

    tray = TrayController()
    assert tray.is_available() is False
    assert tray.ensure_shown() is False
    assert tray._tray is None
    assert tray._menu is None
    assert tray.is_visible is False
    tray.destroy()


def test_tray_controller_unavailable_returns_false(qapp, monkeypatch) -> None:
    monkeypatch.setattr(
        TrayController, "is_available", staticmethod(lambda: False)
    )
    tray = TrayController()
    assert tray.ensure_shown() is False
    assert tray.is_visible is False


def test_tray_controller_show_hide_on_real_tray(qapp) -> None:
    if TrayController.is_headless_platform() or not TrayController.is_available():
        pytest.skip(
            "No real system tray in this environment (typical for offscreen CI)"
        )
    tray = TrayController()
    assert tray.ensure_shown() is True
    assert tray.is_visible is True
    assert tray._tray is not None
    assert tray._tray.toolTip() == TRAY_TOOLTIP
    tray.hide()
    assert tray.is_visible is False
    tray.destroy()
    assert tray._tray is None
    assert tray._menu is None


def test_destroy_is_safe_when_tray_never_constructed(qapp) -> None:
    """destroy() must be a no-op when no native tray objects were built."""
    tray = TrayController()
    if TrayController.is_headless_platform():
        assert tray.ensure_shown() is False
        assert tray._tray is None
    tray.destroy()
    assert tray._tray is None
    assert tray._menu is None
    qapp.processEvents()


def test_forced_tray_destroy_survives_child_process_exit() -> None:
    """
    Subprocess-only regression for construct → destroy → interpreter exit.

    Never build or import QSystemTrayIcon in the main pytest process under
    offscreen — that poisons Qt tray state and SIGSEGVs at suite teardown.
    """
    repo = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import os, sys
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        sys.path.insert(0, {repo!r})
        from PySide6.QtWidgets import QApplication
        from src.presentation.tray_controller import TrayController

        app = QApplication(sys.argv)
        print("child_platform", app.platformName(), flush=True)
        print("child_headless", TrayController.is_headless_platform(), flush=True)
        print("child_controller_available", TrayController.is_available(), flush=True)

        assert TrayController.is_available() is False
        refused = TrayController()
        assert refused.ensure_shown() is False
        assert refused._tray is None
        refused.destroy()
        print("child_refused_ok", flush=True)

        TrayController.is_available = staticmethod(lambda: True)
        tray = TrayController()
        ok = tray.ensure_shown()
        print("child_ensure_shown", ok, flush=True)
        assert ok and tray._tray is not None
        tray.destroy()
        assert tray._tray is None
        app.processEvents()
        print("child_destroy_ok", flush=True)
        raise SystemExit(0)
        """
    ).format(repo=str(repo))

    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    print(completed.stdout)
    print(completed.stderr, file=sys.stderr)
    assert completed.returncode == 0, (
        f"Child exited {completed.returncode} (0x{completed.returncode & 0xFFFFFFFF:08X}); "
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )
    assert "child_refused_ok" in completed.stdout
    assert "child_destroy_ok" in completed.stdout
