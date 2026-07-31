"""Optional smoke: launch a built TileVisionAI.app on macOS Intel."""

from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.qa_e2e


def test_frozen_app_bundle_smoke():
    """
    If TILEVISION_QA_APP_PATH points at TileVisionAI.app (or the binary),
    launch it long enough to prove the customer bundle starts on this Mac.

    Full UI automation of a separate process is optional; the in-process suite
    covers human UI. This test guards packaging/runtime on macOS Intel.
    """
    app_path = os.environ.get("TILEVISION_QA_APP_PATH", "").strip()
    if not app_path:
        pytest.skip("Set TILEVISION_QA_APP_PATH to run frozen .app smoke")

    path = Path(app_path)
    if not path.exists():
        pytest.fail(f"TILEVISION_QA_APP_PATH does not exist: {path}")

    # Prefer --verify-bundle for deterministic smoke without GUI license gate.
    binary = path
    if path.suffix == ".app":
        binary = path / "Contents" / "MacOS" / "TileVisionAI"
    if not binary.exists():
        pytest.fail(f"App binary missing: {binary}")

    env = os.environ.copy()
    env["TILEVISION_DEV_MODE"] = "1"
    proc = subprocess.run(
        [str(binary), "--verify-bundle"],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "bundle OK" in (proc.stdout + proc.stderr)

    # Machine arch note for Intel clients
    machine = platform.machine().lower()
    assert machine in {"x86_64", "i386", "arm64"}, machine
