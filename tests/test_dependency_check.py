"""Tests for dependency_check module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.dependency_check import (
    INSTALL_STEPS,
    PackageSpec,
    all_dependencies_satisfied,
    check_package,
    check_step,
)


def test_install_steps_are_ordered_and_nonempty():
    assert len(INSTALL_STEPS) == 5
    assert INSTALL_STEPS[0].step_id == "foundation"
    assert INSTALL_STEPS[-1].step_id == "extras"


def test_check_package_detects_installed_stdlib_adjacent():
    status = check_package(PackageSpec("pathlib", "pathlib", "pathlib"))
    assert status.installed


def test_extras_step_lists_watchdog():
    extras = INSTALL_STEPS[-1]
    names = [pkg.import_name for pkg in extras.packages]
    assert "watchdog" in names


def test_foundation_step_complete_on_dev_machine():
    status = check_step(INSTALL_STEPS[0])
    assert status.is_complete
