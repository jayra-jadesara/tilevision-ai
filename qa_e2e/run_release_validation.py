#!/usr/bin/env python3
"""
TileVision AI — Release Validation entry point.

Runs the ordered customer validation pipeline on the real stack and writes:

  release_report.html
  release_report.json
  release_report.pdf

Exit code 0 only when EVERY gate and EVERY scenario passes.

Usage:
  export TILEVISION_DEV_MODE=1
  export TILEVISION_QA_OUT=./qa_e2e/artifacts/release_$(date +%Y%m%d_%H%M%S)
  python qa_e2e/run_release_validation.py

Profiles via env:
  TILEVISION_RELEASE_PROFILE=full|pr
    full — S01–S30 with 100 searches + 1800s idle (default for release tags)
    pr   — S01–S30 with reduced stress (10 searches, 60s idle) for PR CI
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# When validating a packaged .app, product modules must come from the frozen
# bundle (sys._MEIPASS). Only append the checkout root so `qa_e2e` resolves;
# never prepend it (that would shadow packaged `src` with source-tree `src`).
if getattr(sys, "frozen", False) or os.environ.get("TILEVISION_QA_PACKAGED_APP") == "1":
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
else:
    sys.path.insert(0, str(ROOT))


def _apply_profile(profile: str) -> None:
    profile = (profile or "full").strip().lower()
    if profile == "pr":
        os.environ.setdefault("TILEVISION_RELEASE_SEARCH_COUNT", "10")
        os.environ.setdefault("TILEVISION_RELEASE_IDLE_SECONDS", "60")
        os.environ.setdefault("TILEVISION_RELEASE_MEM_LOOPS", "4")
    else:
        os.environ.setdefault("TILEVISION_RELEASE_SEARCH_COUNT", "100")
        os.environ.setdefault("TILEVISION_RELEASE_IDLE_SECONDS", "1800")
        os.environ.setdefault("TILEVISION_RELEASE_MEM_LOOPS", "8")


def main() -> int:
    parser = argparse.ArgumentParser(description="TileVision Release Validation")
    parser.add_argument(
        "--out",
        default=os.environ.get(
            "TILEVISION_QA_OUT",
            str(ROOT / "qa_e2e" / "artifacts" / "release_latest"),
        ),
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("TILEVISION_RELEASE_PROFILE", "full"),
        choices=["full", "pr"],
    )
    parser.add_argument(
        "--scenarios",
        default="",
        help="Comma-separated scenario ids (e.g. S01,S02). Default: all.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Use cocoa Qt platform on macOS",
    )
    args = parser.parse_args()

    _apply_profile(args.profile)
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

    scenario_ids = [s.strip().upper() for s in args.scenarios.split(",") if s.strip()] or None

    print("=" * 72)
    print(" TileVision AI — RELEASE VALIDATION")
    print(f" Profile : {args.profile}")
    print(f" Out     : {out}")
    print(f" Qt      : {os.environ.get('QT_QPA_PLATFORM')}")
    print(f" Packaged: {os.environ.get('TILEVISION_QA_PACKAGED_APP', '0')}")
    print(f" Frozen  : {bool(getattr(sys, 'frozen', False))}")
    print(" Policy  : PASS only if ALL gates and ALL scenarios pass")
    print(" Mocks   : FORBIDDEN")
    print("=" * 72)

    from qa_e2e.release.pipeline import run_release_validation

    work = Path(tempfile.mkdtemp(prefix="tilevision_release_"))
    payload = run_release_validation(work_dir=work, out_dir=out, scenario_ids=scenario_ids)

    print()
    print(f"VERDICT: {payload['verdict']}")
    print(
        f"Gates: {payload['gates_passed']}/{payload['gates_total']}  "
        f"Scenarios: {payload['scenarios_passed']}/{payload['scenarios_total']}  "
        f"Duration: {payload['duration_s']:.1f}s"
    )
    for key, path in payload.get("reports", {}).items():
        print(f"  {key}: {path}")

    failed = [s for s in payload.get("scenarios", []) if not s.get("ok")]
    if failed:
        print("\nFailed scenarios:")
        for s in failed:
            line = f"  - {s.get('id')} {s.get('name')}: {s.get('error') or s.get('detail')}"
            # Windows cp1252 consoles cannot encode arrows / fancy dashes.
            try:
                print(line)
            except UnicodeEncodeError:
                print(line.encode("ascii", "replace").decode("ascii"))

    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    # Prefer UTF-8 on Windows so report printing cannot crash after a real FAIL.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    code = main()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    # Windows + Qt teardown under Git Bash often reports exit 127 after a real
    # PASS (atexit / QApplication shutdown). Hard-exit so CI sees the verdict.
    if sys.platform == "win32":
        os._exit(code)
    raise SystemExit(code)
