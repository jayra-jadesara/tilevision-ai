"""Unit tests for CI pytest crash recovery helpers."""

from __future__ import annotations

from pathlib import Path

from scripts.run_ci_pytest import (
    _is_native_teardown_crash,
    _is_windows_crash,
    _junit_green,
)


def test_native_teardown_crash_codes_are_detected():
    assert _is_native_teardown_crash(3221225477)  # unsigned STATUS_ACCESS_VIOLATION
    assert _is_native_teardown_crash(-1073741819)  # signed form
    assert _is_native_teardown_crash(127)
    assert _is_native_teardown_crash(139)  # Linux SIGSEGV
    assert _is_native_teardown_crash(-11)  # SIGSEGV as negative
    assert not _is_native_teardown_crash(0)
    assert not _is_native_teardown_crash(1)
    # Alias kept for older callers/tests.
    assert _is_windows_crash(139)


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
