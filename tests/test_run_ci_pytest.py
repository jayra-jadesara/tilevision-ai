"""Unit tests for Windows CI pytest crash recovery helpers."""

from __future__ import annotations

from pathlib import Path

from scripts.run_ci_pytest import _is_windows_crash, _junit_green


def test_windows_access_violation_codes_are_detected():
    assert _is_windows_crash(3221225477)  # unsigned STATUS_ACCESS_VIOLATION
    assert _is_windows_crash(-1073741819)  # signed form
    assert _is_windows_crash(127)
    assert _is_windows_crash(139)
    assert not _is_windows_crash(0)
    assert not _is_windows_crash(1)


def test_junit_green_requires_finished_suite(tmp_path: Path):
    good = tmp_path / "ok.xml"
    good.write_text('<testsuite tests="4" failures="0" errors="0"></testsuite>')
    assert _junit_green(good)

    bad = tmp_path / "bad.xml"
    bad.write_text('<testsuite tests="4" failures="1" errors="0"></testsuite>')
    assert not _junit_green(bad)

    empty = tmp_path / "empty.xml"
    empty.write_text('<testsuite tests="0" failures="0" errors="0"></testsuite>')
    assert not _junit_green(empty)
