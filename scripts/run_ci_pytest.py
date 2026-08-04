#!/usr/bin/env python3
"""
CI pytest runner with Windows Qt/Git-Bash crash mitigation.

PySide teardown under Git Bash on windows-latest frequently kills the
pytest process with NTSTATUS access-violation (0xC0000005 → 3221225477)
or Bash-mapped 127/139 — often after a fully green suite. This wrapper:
  1. Runs pytest with junitxml in a subprocess
  2. Treats known crash exit codes as success when junit is green
  3. Retries once on Windows when junit is missing/incomplete
  4. Hard-exits the wrapper with a clamped 0/1 code (no Qt loaded here)
"""

from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUNIT = ROOT / "pytest-results.xml"

# STATUS_ACCESS_VIOLATION and friends as returned by subprocess on Windows.
_STATUS_ACCESS_VIOLATION = 0xC0000005


def _is_windows_crash(code: int) -> bool:
    if code in (127, 139):
        return True
    # subprocess may surface NTSTATUS as a large unsigned or negative signed int.
    unsigned = code & 0xFFFFFFFF
    if unsigned == _STATUS_ACCESS_VIOLATION:
        return True
    # Other NT failure statuses (0xCxxxxxxx)
    if unsigned >= 0xC0000000:
        return True
    return False


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


def _run_pytest(markers: str) -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-q",
        "--tb=short",
        "-m",
        markers,
        f"--junitxml={JUNIT}",
    ]
    print("+", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=str(ROOT))
    return int(completed.returncode)


def main() -> int:
    markers = os.environ.get("TILEVISION_PYTEST_MARKERS", "not slow")
    is_windows = sys.platform == "win32"
    attempts = 2 if is_windows else 1
    status = 1
    for attempt in range(1, attempts + 1):
        status = _run_pytest(markers)
        if status == 0:
            break
        if is_windows and _is_windows_crash(status) and _junit_green(JUNIT):
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

    # Wrapper does not import Qt; clamp to 0/1 so os._exit never overflows.
    final = 0 if status == 0 else 1
    if is_windows:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(final)
    return final


if __name__ == "__main__":
    raise SystemExit(main())
