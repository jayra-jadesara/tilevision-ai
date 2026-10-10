#!/usr/bin/env python3
"""
Scripted mid-session license expiry cutover demo (no full AI warm-up).

Simulates: valid license → periodic re-check flips to invalid → monitoring
stopped, notifications suppressed, main window hidden, shared LicenseView
path invoked.

Usage:
  QT_QPA_PLATFORM=offscreen python scripts/demo_license_cutover.py

Override re-check interval (ms) via TILEVISION_LICENSE_RECHECK_MS if exercising
the real app instead of this harness.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
from PySide6.QtWidgets import QApplication

from src.core.use_cases.monitor_folder import FolderMonitorController, is_watchdog_available
from src.presentation.license_gate import LicenseSessionGuard, show_license_activation
from src.presentation.tray_controller import TrayController


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(90, 40, 20)).save(path, format="PNG")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("demo.license_cutover")

    app = QApplication.instance() or QApplication(sys.argv)

    validate = MagicMock()
    # First check valid; subsequent checks invalid (simulates expiry while open).
    validate.verify_existing_license.side_effect = [
        {"customer_name": "Demo Co", "is_trial": True, "days_remaining": 0},
        None,
        None,
    ]

    use_case = MagicMock()
    use_case.index_changed_file.return_value = 1
    notifications_enabled = {"value": True}
    events: list = []

    def _callback(path, action, success, message):
        if not notifications_enabled["value"]:
            log.info("SUPPRESSED notification for %s (%s)", Path(path).name, action)
            return
        events.append((path, action, success))
        log.info("Notification: %s → %s", Path(path).name, action)

    folder_monitor = None
    if is_watchdog_available():
        folder_monitor = FolderMonitorController(
            indexing_use_case=use_case,
            on_file_indexed_callback=_callback,
        )
    else:
        folder_monitor = MagicMock()
        folder_monitor.is_running = True
        folder_monitor.stop_monitoring = MagicMock()

    tray = TrayController()
    # Headless: tray usually unavailable; still exercise hide() path.
    tray_available = tray.ensure_shown()
    log.info("BEFORE cutover: tray_available=%s tray_visible=%s", tray_available, tray.is_visible)

    main_window = MagicMock()
    main_window.isVisible = MagicMock(return_value=True)
    hidden = {"value": False}

    def _hide():
        hidden["value"] = True
        main_window.isVisible.return_value = False
        log.info("MainWindow.hide() called")

    main_window.hide.side_effect = _hide
    main_window.cancel_active_indexing_for_license_cutover = MagicMock()

    with tempfile.TemporaryDirectory() as tmp:
        watch = Path(tmp) / "watched"
        _write_png(watch / "tile.png")
        if is_watchdog_available() and isinstance(folder_monitor, FolderMonitorController):
            folder_monitor.start_monitoring([str(watch)])
            log.info(
                "BEFORE cutover: folder_monitor.is_running=%s",
                folder_monitor.is_running,
            )
        else:
            log.info("BEFORE cutover: using mock folder_monitor (watchdog unavailable)")

        license_shown = {"value": False}

        def _cutover() -> None:
            log.info("=== CUTOVER START ===")
            notifications_enabled["value"] = False
            if folder_monitor is not None:
                folder_monitor.stop_monitoring()
                log.info(
                    "AFTER stop_monitoring: is_running=%s",
                    getattr(folder_monitor, "is_running", "n/a"),
                )
            tray.hide()
            log.info("AFTER tray.hide: tray_visible=%s", tray.is_visible)
            main_window.cancel_active_indexing_for_license_cutover()
            main_window.hide()
            log.info("AFTER main_window.hide: visible=%s", main_window.isVisible())

            with patch("src.presentation.license_gate.LicenseView") as lv_cls:
                dialog = MagicMock()
                dialog.is_activated = False
                dialog.exec = MagicMock(
                    side_effect=lambda: log.info("LicenseView.exec() — license-only screen")
                )
                lv_cls.return_value = dialog
                renewed = show_license_activation(validate, theme="light")
            license_shown["value"] = True
            log.info(
                "License gate result: renewed=%s (expected None when user skips)",
                renewed,
            )
            log.info("=== CUTOVER END ===")

        guard = LicenseSessionGuard(validate, interval_ms=60_000)
        guard.license_invalidated.connect(_cutover)

        # Simulate mid-session: first recheck still valid, second is expired.
        log.info("Recheck #1 (still valid) via LicenseSessionGuard.recheck_now()...")
        guard.start()
        guard.recheck_now()
        log.info(
            "After recheck #1: guard_active=%s notifications=%s main_visible=%s",
            guard.is_active,
            notifications_enabled["value"],
            main_window.isVisible(),
        )

        log.info("Recheck #2 (expired) via LicenseSessionGuard.recheck_now()...")
        guard.recheck_now()

        assert hidden["value"] is True, "Main window should be hidden after cutover"
        assert license_shown["value"] is True, "License screen path must run"
        assert notifications_enabled["value"] is False, "Notifications must be suppressed"
        if isinstance(folder_monitor, FolderMonitorController):
            assert folder_monitor.is_running is False, "Monitoring must stop"
        else:
            folder_monitor.stop_monitoring.assert_called()

        log.info("DEMO OK — cutover stopped monitoring, hid main UI, showed license gate.")
        tray.destroy()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
