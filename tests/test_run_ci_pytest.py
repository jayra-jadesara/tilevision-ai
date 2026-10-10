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


def test_linux_wrapper_does_not_mask_sigsegv(tmp_path: Path, monkeypatch) -> None:
    """Linux green-junit + SIGSEGV must stay a failure (fix the real crash)."""
    import scripts.run_ci_pytest as runner

    junit = tmp_path / "pytest-results.xml"
    junit.write_text('<testsuite tests="10" failures="0" errors="0"></testsuite>')
    monkeypatch.setattr(runner, "JUNIT", junit)
    monkeypatch.setattr(runner, "JUNIT_TRAY", tmp_path / "tray.xml")
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner, "_run_main_suite", lambda markers: -11)

    assert runner.main() == 1


def test_wrapper_runs_tray_suite_after_main(tmp_path: Path, monkeypatch) -> None:
    import scripts.run_ci_pytest as runner

    calls: list[str] = []

    def _main(markers: str) -> int:
        calls.append("main")
        return 0

    def _tray(markers: str) -> int:
        calls.append("tray")
        return 0

    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner, "_run_main_suite", _main)
    monkeypatch.setattr(runner, "_run_tray_suite", _tray)

    assert runner.main() == 0
    assert calls == ["main", "tray"]


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
