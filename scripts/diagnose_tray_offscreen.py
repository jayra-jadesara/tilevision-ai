#!/usr/bin/env python3
"""
Print tray availability under the current QT_QPA_PLATFORM.

Used to confirm Task 1 on CI: platformName, whether TrayController refuses
construction, and that we do not probe isSystemTrayAvailable() under headless
QPA (that probe itself is unsafe on some ubuntu-latest offscreen images).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from src.presentation.tray_controller import TrayController

    app = QApplication(sys.argv)
    platform = app.platformName()
    headless = TrayController.is_headless_platform()
    available = TrayController.is_available()
    print(f"QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM')!r}")
    print(f"platformName={platform!r}")
    print(f"is_headless_platform={headless}")
    print(f"TrayController.is_available={available}")
    # Intentionally do not call QSystemTrayIcon.isSystemTrayAvailable() when
    # headless — that probe can initialize a broken tray backend on Linux CI.
    if not headless:
        from PySide6.QtWidgets import QSystemTrayIcon

        print(f"isSystemTrayAvailable={QSystemTrayIcon.isSystemTrayAvailable()}")
    else:
        print("isSystemTrayAvailable=skipped_probe_under_headless")

    tray = TrayController()
    shown = tray.ensure_shown()
    print(f"ensure_shown={shown} tray_constructed={tray._tray is not None}")
    tray.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
