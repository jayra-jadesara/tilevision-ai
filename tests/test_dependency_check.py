"""Tests for dependency_check module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.dependency_check import (
    INSTALL_STEPS,
    REQUIRED_STEP_COUNT,
    PackageSpec,
    all_dependencies_satisfied,
    check_all_steps,
    check_package,
    check_step,
    step_is_complete,
)


def test_install_steps_include_runtime_sqlite_and_gpu_optional():
    assert len(INSTALL_STEPS) == REQUIRED_STEP_COUNT + 1
    assert INSTALL_STEPS[0].step_id == "runtime"
    assert INSTALL_STEPS[-1].step_id == "gpu_optional"
    assert INSTALL_STEPS[-1].optional is True


def test_runtime_step_checks_python_and_sqlite():
    runtime = check_step(INSTALL_STEPS[0])
    assert runtime.is_complete
    names = [pkg.spec.import_name for pkg in runtime.packages]
    assert names == ["__python__", "__sqlite__"]


def test_check_package_detects_installed_stdlib_adjacent():
    status = check_package(PackageSpec("pathlib", "pathlib", "pathlib"))
    assert status.installed


def test_extras_step_lists_watchdog_as_required():
    extras = next(step for step in INSTALL_STEPS if step.step_id == "extras")
    watchdog = next(pkg for pkg in extras.packages if pkg.import_name == "watchdog")
    assert watchdog.optional is False


def test_ai_step_lists_tokenizers():
    ai = next(step for step in INSTALL_STEPS if step.step_id == "ai_core")
    names = [pkg.import_name for pkg in ai.packages]
    assert "tokenizers" in names


def test_foundation_step_complete_on_dev_machine():
    foundation = next(step for step in INSTALL_STEPS if step.step_id == "foundation")
    status = check_step(foundation)
    assert status.is_complete


def test_gpu_optional_step_auto_complete_without_nvidia():
    gpu = check_step(INSTALL_STEPS[-1])
    assert step_is_complete(gpu)


def test_all_dependencies_satisfied_ignores_optional_gpu():
    for status in check_all_steps():
        if status.step.optional:
            continue
        assert step_is_complete(status)
