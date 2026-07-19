#!/usr/bin/env python3
"""
Cross-platform preflight check before running or packaging TileVision AI.

Usage:
    python scripts/preflight_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.model_paths import model_status_message  # noqa: E402
from src.utils.dependency_check import all_dependencies_satisfied, check_all_steps  # noqa: E402
from src.utils.platform_info import (  # noqa: E402
    detect_display_adapters,
    is_linux,
    is_macos,
    is_windows,
)


def _check_python() -> tuple[bool, str]:
    if sys.version_info >= (3, 12):
        return True, f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return False, f"Python 3.12+ required (found {sys.version_info.major}.{sys.version_info.minor})"


def main() -> int:
    print("=== TileVision AI — Preflight Check ===")
    print()

    ok, msg = _check_python()
    print(f"[{'OK' if ok else 'FAIL'}] {msg}")
    if not ok:
        return 1

    platform_name = "Windows" if is_windows() else "macOS" if is_macos() else "Linux"
    print(f"[OK ] Platform: {platform_name} ({sys.platform})")

    adapters = detect_display_adapters()
    if adapters:
        print(f"[OK ] Graphics: {', '.join(adapters[:3])}")
    else:
        print("[INFO] Graphics: could not detect adapters (CPU mode is fine)")

    deps_ok = all_dependencies_satisfied()
    print(f"[{'OK' if deps_ok else 'FAIL'}] Python packages: {'all required packages installed' if deps_ok else 'missing packages'}")
    if not deps_ok:
        for status in check_all_steps():
            if status.step.optional:
                continue
            for pkg in status.packages:
                if not pkg.installed:
                    print(f"       - missing: {pkg.spec.display_name}")

    model_msg = model_status_message()
    model_ok = model_msg.startswith("Ready") or model_msg.startswith("Will download")
    print(f"[{'OK' if model_ok else 'FAIL'}] DINOv2 model: {model_msg}")

    if is_linux():
        print()
        print("Linux tip: run  bash scripts/check_qt_deps.sh  if the UI fails to start.")

    print()
    if deps_ok and model_ok:
        print("Preflight passed. Launch with:  python main.py")
        return 0

    print("Preflight found issues — fix the items above before launching.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
