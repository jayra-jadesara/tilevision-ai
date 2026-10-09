"""Tests for minimize-to-tray state transitions (headless/offscreen safe)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QSystemTrayIcon

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


def test_close_event_accepts_when_tray_disabled(
    qapp, tmp_path: Path, catalogue_master_service
) -> None:
    window = _make_main_window(tmp_path, catalogue_master_service, minimize=False)
    window.show()

    event = QCloseEvent()
    window.closeEvent(event)

    assert event.isAccepted() is True
    window.deleteLater()


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


def test_restore_from_tray(qapp, tmp_path: Path, catalogue_master_service) -> None:
    window = _make_main_window(tmp_path, catalogue_master_service, minimize=True)
    window.show()
    window.hide()
    assert not window.isVisible()
    window.restore_from_tray()
    assert window.isVisible()
    window.deleteLater()


def test_tray_controller_unavailable_returns_false(qapp, monkeypatch) -> None:
    monkeypatch.setattr(
        QSystemTrayIcon, "isSystemTrayAvailable", staticmethod(lambda: False)
    )
    tray = TrayController()
    assert tray.ensure_shown() is False
    assert tray.is_visible is False


def test_tray_controller_show_hide_on_real_tray(qapp) -> None:
    # Must not call isSystemTrayAvailable() at module import — that can
    # segfault before QApplication exists on some Linux/offscreen hosts.
    if not QSystemTrayIcon.isSystemTrayAvailable():
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
