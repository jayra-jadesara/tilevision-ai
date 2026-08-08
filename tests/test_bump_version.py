"""Tests for scripts/bump_version.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import bump_version as bv

ROOT = Path(__file__).resolve().parents[1]

SAMPLE_VERSION_PY = 'APP_VERSION = "1.2.33"\n'
SAMPLE_SETUP_ISS = '#define MyAppVersion "1.2.33"\n'
SAMPLE_ADMIN_ISS = '#define MyAppVersion "1.2.32"\n'


def _write_version_files(root: Path, app: str, setup: str, admin: str) -> None:
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "packaging").mkdir(parents=True, exist_ok=True)
    (root / "src" / "version.py").write_text(f'APP_VERSION = "{app}"\n', encoding="utf-8")
    (root / "packaging" / "tilevision_setup.iss").write_text(
        f'#define MyAppVersion "{setup}"\n',
        encoding="utf-8",
    )
    (root / "packaging" / "tilevision_admin_setup.iss").write_text(
        f'#define MyAppVersion "{admin}"\n',
        encoding="utf-8",
    )


def test_validate_version_string_rejects_malformed() -> None:
    for bad in ("", "1.2", "abc", "1.2.3.4", "v1.2.3"):
        with pytest.raises(ValueError, match="Invalid version"):
            bv.validate_version_string(bad)


def test_validate_version_string_accepts_patch_and_suffix() -> None:
    bv.validate_version_string("1.2.34")
    bv.validate_version_string("1.2.34-rc1")


def test_bump_updates_all_three_files(tmp_path: Path) -> None:
    _write_version_files(tmp_path, "1.2.33", "1.2.33", "1.2.32")

    changes = bv.bump_version_files("1.2.34", root=tmp_path)

    assert len(changes) == 3
    assert bv.read_current_version(tmp_path) == "1.2.34"
    assert bv.extract_version("src/version.py", (tmp_path / "src" / "version.py").read_text()) == "1.2.34"
    assert (
        bv.extract_version(
            "packaging/tilevision_setup.iss",
            (tmp_path / "packaging" / "tilevision_setup.iss").read_text(),
        )
        == "1.2.34"
    )
    assert (
        bv.extract_version(
            "packaging/tilevision_admin_setup.iss",
            (tmp_path / "packaging" / "tilevision_admin_setup.iss").read_text(),
        )
        == "1.2.34"
    )


def test_bump_reports_write_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_version_files(tmp_path, "1.2.33", "1.2.33", "1.2.33")
    real_write_text = Path.write_text
    calls = {"count": 0}

    def flaky_write(self: Path, data: str, encoding: str | None = None) -> int:
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated write failure")
        return real_write_text(self, data, encoding=encoding or "utf-8")

    monkeypatch.setattr(Path, "write_text", flaky_write)

    with pytest.raises(RuntimeError, match="Failed to write"):
        bv.bump_version_files("1.2.34", root=tmp_path)

    assert bv.read_current_version(tmp_path) == "1.2.34"
    assert (
        bv.extract_version(
            "packaging/tilevision_setup.iss",
            (tmp_path / "packaging" / "tilevision_setup.iss").read_text(),
        )
        == "1.2.33"
    )


def test_main_exits_nonzero_on_partial_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_version_files(tmp_path, "1.2.33", "1.2.33", "1.2.33")
    real_write_text = Path.write_text
    calls = {"count": 0}

    def flaky_write(self: Path, data: str, encoding: str | None = None) -> int:
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated write failure")
        return real_write_text(self, data, encoding=encoding or "utf-8")

    monkeypatch.setattr(Path, "write_text", flaky_write)

    assert bv.main(["1.2.34"], root=tmp_path) == 1


def test_main_rejects_malformed_version(tmp_path: Path) -> None:
    _write_version_files(tmp_path, "1.2.33", "1.2.33", "1.2.33")

    assert bv.main(["1.2"], root=tmp_path) == 1
    assert bv.read_current_version(tmp_path) == "1.2.33"


def test_main_propagates_validate_ci_build_exit_code(tmp_path: Path) -> None:
    _write_version_files(tmp_path, "1.2.33", "1.2.33", "1.2.33")

    with patch.object(bv, "run_validate_ci_build", return_value=1):
        assert bv.main(["1.2.34"], root=tmp_path) == 1


def test_main_runs_validate_ci_build(tmp_path: Path) -> None:
    _write_version_files(tmp_path, "1.2.33", "1.2.33", "1.2.33")
    validate_script = tmp_path / "scripts" / "validate_ci_build.py"
    validate_script.parent.mkdir(parents=True, exist_ok=True)
    validate_script.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "ROOT = Path(__file__).resolve().parents[1]\n"
        "app = (ROOT / 'src' / 'version.py').read_text()\n"
        "iss = (ROOT / 'packaging' / 'tilevision_setup.iss').read_text()\n"
        "admin = (ROOT / 'packaging' / 'tilevision_admin_setup.iss').read_text()\n"
        "v = app.split('\"')[1]\n"
        "if v not in iss or v not in admin:\n"
        "    print('mismatch', file=sys.stderr); sys.exit(1)\n"
        "print('ok')\n",
        encoding="utf-8",
    )

    assert bv.main(["1.2.34"], root=tmp_path) == 0


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "bump_version.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "MAJOR.MINOR.PATCH" in result.stdout
