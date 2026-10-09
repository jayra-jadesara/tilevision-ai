#!/usr/bin/env python3
"""
CI pytest runner with Windows Qt/Git-Bash crash mitigation.

PySide teardown under Git Bash on windows-latest frequently kills the
pytest process with NTSTATUS access-violation (0xC0000005 → 3221225477)
or Bash-mapped 127/139 — often after a fully green suite. This wrapper:
  1. Runs the main pytest suite (excluding tray minimize tests) with junitxml
  2. Runs tray minimize tests in a *fresh* interpreter so QSystemTrayIcon
     never shares process teardown with faiss/torch/full-suite Qt state
     (that combination SIGSEGVs after green on ubuntu-latest / macos-15)
  3. On Windows only, treats known crash exit codes as success when junit
     is green
  4. Retries once on Windows when junit is missing/incomplete
  5. Hard-exits the wrapper with a clamped 0/1 code (no Qt loaded here)
"""

from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUNIT = ROOT / "pytest-results.xml"
JUNIT_TRAY = ROOT / "pytest-results-tray.xml"
TRAY_TESTS = "tests/test_tray_minimize.py"

# STATUS_ACCESS_VIOLATION and friends as returned by subprocess on Windows.
_STATUS_ACCESS_VIOLATION = 0xC0000005


def _is_native_teardown_crash(code: int) -> bool:
    """True for process-killing native crashes that can follow a green suite."""
    if code in (127, 139, -11):
        # 139 / -11 = SIGSEGV; 127 sometimes from Bash after a killed child.
        return True
    # subprocess may surface NTSTATUS as a large unsigned or negative signed int.
    unsigned = code & 0xFFFFFFFF
    if unsigned == _STATUS_ACCESS_VIOLATION:
        return True
    # Other NT failure statuses (0xCxxxxxxx)
    if unsigned >= 0xC0000000:
        return True
    return False


# Back-compat alias for tests / importers.
_is_windows_crash = _is_native_teardown_crash


def _junit_green(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        root = ET.parse(path).getroot()
    except Exception as exc:
        print(f"junit parse failed: {exc}", file=sys.stderr)
        return False
    suites = root.findall("testsuite")
    if not suites and root.tag == "testsuite":
        suites = [root]
    failures = errors = tests = 0
    for suite in suites:
        failures += int(suite.attrib.get("failures", 0))
        errors += int(suite.attrib.get("errors", 0))
        tests += int(suite.attrib.get("tests", 0))
    print(f"junit: tests={tests} failures={failures} errors={errors}", flush=True)
    return tests > 0 and failures == 0 and errors == 0


def _run_pytest(markers: str, *extra: str, junit: Path = JUNIT) -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "-m",
        markers,
        f"--junitxml={junit}",
        *extra,
    ]
    print("+", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(ROOT))
    return int(completed.returncode)


def _run_main_suite(markers: str) -> int:
    # Keep tray minimize tests out of the faiss/torch-heavy process.
    return _run_pytest(
        markers,
        "tests/",
        f"--ignore={TRAY_TESTS}",
        junit=JUNIT,
    )


def _run_tray_suite(markers: str) -> int:
    print(
        "Running tray minimize tests in a fresh interpreter "
        "(avoids offscreen QSystemTrayIcon teardown SIGSEGV after full suite)",
        flush=True,
    )
    return _run_pytest(markers, TRAY_TESTS, junit=JUNIT_TRAY)


def main() -> int:
    markers = os.environ.get("TILEVISION_PYTEST_MARKERS", "not slow")
    is_windows = sys.platform == "win32"
    attempts = 2 if is_windows else 1
    status = 1
    for attempt in range(1, attempts + 1):
        status = _run_main_suite(markers)
        if status == 0:
            break
        # Windows-only: green junit + native crash → success.
        if is_windows and _is_native_teardown_crash(status) and _junit_green(JUNIT):
            print(
                f"Windows pytest exited {status} (0x{status & 0xFFFFFFFF:08X}) "
                "after green junit — treating as success",
                flush=True,
            )
            status = 0
            break
        if is_windows and attempt < attempts:
            print(
                f"Windows pytest exited {status} (0x{status & 0xFFFFFFFF:08X}) — retrying once",
                flush=True,
            )
            continue
        break

    if status != 0:
        final = 1
    else:
        tray_status = _run_tray_suite(markers)
        if tray_status != 0 and is_windows and _is_native_teardown_crash(tray_status):
            if _junit_green(JUNIT_TRAY):
                print(
                    f"Windows tray pytest exited {tray_status} "
                    f"(0x{tray_status & 0xFFFFFFFF:08X}) after green junit — "
                    "treating as success",
                    flush=True,
                )
                tray_status = 0
        if tray_status != 0:
            print(f"Tray pytest suite failed with exit {tray_status}", flush=True)
            final = 1
        else:
            final = 0

    if is_windows:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(final)
    return final


if __name__ == "__main__":
    raise SystemExit(main())
