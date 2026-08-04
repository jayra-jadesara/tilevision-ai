#!/usr/bin/env python3
"""Return 0 if a pytest JUnit XML report has tests and zero failures/errors."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: ci_junit_ok.py <pytest-results.xml>", file=sys.stderr)
        return 2
    path = argv[1]
    root = ET.parse(path).getroot()
    suites = root.findall("testsuite")
    if not suites and root.tag == "testsuite":
        suites = [root]
    failures = errors = tests = 0
    for suite in suites:
        failures += int(suite.attrib.get("failures", 0))
        errors += int(suite.attrib.get("errors", 0))
        tests += int(suite.attrib.get("tests", 0))
    print(f"junit recovery: tests={tests} failures={failures} errors={errors}")
    if tests <= 0 or failures or errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
