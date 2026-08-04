#!/usr/bin/env python3
"""
CI pytest runner with Windows Qt/Git-Bash crash mitigation.

PySide teardown under Git Bash on windows-latest frequently exits 127/139
after a fully green suite. This wrapper:
  1. Runs pytest with junitxml
  2. Retries once on Windows when the process crashes
  3. Treats 127/139 as success when junit shows a finished green suite
  4. Hard-exits so the wrapper itself does not hit the same teardown
"""

from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUNIT = ROOT / "pytest-results.xml"


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
    print(f"junit: tests={tests} failures={failures} errors={errors}")
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
        if is_windows and status in (127, 139) and _junit_green(JUNIT):
            print(
                f"Windows pytest exited {status} after green junit — treating as success",
                flush=True,
            )
            status = 0
            break
        if is_windows and attempt < attempts:
            print(f"Windows pytest exited {status} — retrying once", flush=True)
            continue
        break

    # Avoid wrapper-process Qt teardown oddities on Windows Git Bash.
    if is_windows:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
