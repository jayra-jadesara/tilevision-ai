#!/usr/bin/env python3
"""
Run the TileVision AI human-like E2E QA suite and emit an HTML report.

Usage (macOS Intel showroom machine or GitHub macos-13):

    export TILEVISION_DEV_MODE=1
    export TILEVISION_QA_OUT=./qa_e2e/artifacts/run_001
    # optional: path to built app for frozen smoke
    # export TILEVISION_QA_APP_PATH=/path/to/TileVisionAI.app

    python qa_e2e/run_qa.py

Environment:
    TILEVISION_QA_OUT     Report + screenshot directory
    TILEVISION_QA_SEED    Human RNG seed (default 42)
    TILEVISION_QA_SPEED   >1 faster, <1 slower human timing
    TILEVISION_QA_TILES   Catalogue size (default 12)
    TILEVISION_QA_APP_PATH  Optional TileVisionAI.app for smoke
    QT_QPA_PLATFORM       offscreen (CI) or cocoa (interactive Mac)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="TileVision human E2E QA runner")
    parser.add_argument(
        "--out",
        default=os.environ.get("TILEVISION_QA_OUT", str(ROOT / "qa_e2e" / "artifacts" / "latest")),
        help="Artifact / HTML report directory",
    )
    parser.add_argument(
        "-k",
        default=None,
        help="pytest -k expression",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Use cocoa Qt platform (real macOS windows) instead of offscreen",
    )
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["TILEVISION_QA_OUT"] = str(out)
    os.environ.setdefault("TILEVISION_DEV_MODE", "1")
    os.environ.setdefault("TILEVISION_LOG_LEVEL", "INFO")
    os.environ.setdefault("TILEVISION_PROFILE", "1")
    if args.interactive:
        os.environ["QT_QPA_PLATFORM"] = "cocoa"
    else:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import pytest

    pytest_args = [
        str(ROOT / "qa_e2e" / "scenarios"),
        "-v",
        "--tb=short",
        "-m",
        "qa_e2e",
        f"--junitxml={out / 'junit.xml'}",
    ]
    if args.k:
        pytest_args.extend(["-k", args.k])

    print("=" * 60)
    print(" TileVision AI — Human E2E QA")
    print(f" Artifacts: {out}")
    print(f" Qt platform: {os.environ.get('QT_QPA_PLATFORM')}")
    print("=" * 60)
    code = pytest.main(pytest_args)
    report = out / "qa_report.html"
    if report.exists():
        print(f"\nHTML report: {report}")
    else:
        print("\nWARNING: HTML report was not written (session fixture may have skipped).")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
