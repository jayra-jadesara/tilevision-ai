"""Tests for mid-session license re-check and expiry cutover routing."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from src.presentation.license_gate import (
    LicenseSessionGuard,
    resolve_license_recheck_interval_ms,
    show_license_activation,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    return app


def test_resolve_license_recheck_interval_env_override(monkeypatch) -> None:
    monkeypatch.setenv("TILEVISION_LICENSE_RECHECK_MS", "5000")
    assert resolve_license_recheck_interval_ms() == 5000

    monkeypatch.setenv("TILEVISION_LICENSE_RECHECK_MS", "not-a-number")
    assert resolve_license_recheck_interval_ms(default_ms=9000) == 9000

    monkeypatch.setenv("TILEVISION_LICENSE_RECHECK_MS", "100")
    assert resolve_license_recheck_interval_ms(default_ms=9000) == 9000


def test_license_guard_emits_when_verify_returns_none(qapp) -> None:
    validate = MagicMock()
    validate.verify_existing_license.side_effect = [
        {"customer_name": "Acme", "is_trial": True},
        None,
    ]

    guard = LicenseSessionGuard(validate, interval_ms=60_000)
    try:
        invalidated = []
        guard.license_invalidated.connect(lambda: invalidated.append(True))

        guard.recheck_now()
        assert invalidated == []
        assert guard.is_active is False  # not started yet

        guard.start()
        assert guard.is_active

        guard.recheck_now()
        assert invalidated == [True]
        assert guard.is_active is False  # stopped on invalidation
    finally:
        guard.stop()
        guard.deleteLater()
        qapp.processEvents()


def test_license_guard_cutover_stops_monitoring_and_shows_license(qapp) -> None:
    """
    Simulate the app cutover handler: when the guard fires, monitoring stops
    and the shared license activation path is invoked (not the main window).
    """
    validate = MagicMock()
    validate.verify_existing_license.return_value = None

    folder_monitor = MagicMock()
    tray = MagicMock()
    main_window = MagicMock()
    notifications = {"enabled": True}
    routed = {"license_shown": False, "main_still_shown": True}

    def _cutover() -> None:
        notifications["enabled"] = False
        folder_monitor.stop_monitoring()
        tray.hide()
        main_window.cancel_active_indexing_for_license_cutover()
        main_window.hide()
        routed["main_still_shown"] = False
        # Shared gate path (startup + mid-session) — stubbed as "not renewed".
        with patch(
            "src.presentation.license_gate.LicenseView",
        ) as license_view_cls:
            dialog = MagicMock()
            dialog.is_activated = False
            dialog.exec = MagicMock()
            license_view_cls.return_value = dialog
            renewed = show_license_activation(validate, theme="light")
        assert renewed is None
        routed["license_shown"] = True

    guard = LicenseSessionGuard(validate, interval_ms=60_000)
    try:
        guard.license_invalidated.connect(_cutover)
        guard.start()
        guard.recheck_now()

        folder_monitor.stop_monitoring.assert_called_once()
        tray.hide.assert_called_once()
        main_window.hide.assert_called_once()
        assert notifications["enabled"] is False
        assert routed["license_shown"] is True
        assert routed["main_still_shown"] is False
    finally:
        guard.stop()
        guard.deleteLater()
        qapp.processEvents()


def test_show_license_activation_returns_details_when_activated(qapp, monkeypatch) -> None:
    validate = MagicMock()
    validate.verify_existing_license.return_value = {
        "customer_name": "Renewed Co",
        "is_trial": False,
    }

    fake_dialog = MagicMock()
    fake_dialog.is_activated = True
    fake_dialog.exec = MagicMock()

    with patch("src.presentation.license_gate.LicenseView", return_value=fake_dialog):
        details = show_license_activation(validate, theme="dark")

    assert details["customer_name"] == "Renewed Co"
    fake_dialog.exec.assert_called_once()


def test_show_license_activation_returns_none_when_skipped(qapp, monkeypatch) -> None:
    validate = MagicMock()
    fake_dialog = MagicMock()
    fake_dialog.is_activated = False
    fake_dialog.exec = MagicMock()

    with patch("src.presentation.license_gate.LicenseView", return_value=fake_dialog):
        details = show_license_activation(validate, theme="light")

    assert details is None
    validate.verify_existing_license.assert_not_called()
