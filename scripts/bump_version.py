#!/usr/bin/env python3
"""Bump version across all hardcoded version files in one atomic pass."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:-[A-Za-z0-9.]+)?$")

APP_VERSION_RE = re.compile(r'^APP_VERSION = "([^"]+)"', re.MULTILINE)
MY_APP_VERSION_RE = re.compile(r'#define MyAppVersion "([^"]+)"')

VERSION_FILES: tuple[str, ...] = (
    "src/version.py",
    "packaging/tilevision_setup.iss",
    "packaging/tilevision_admin_setup.iss",
)


@dataclass(frozen=True)
class FileChange:
    rel_path: str
    old_version: str
    new_version: str


def validate_version_string(version: str) -> None:
    if not version or not VERSION_PATTERN.fullmatch(version):
        raise ValueError(
            f"Invalid version {version!r}: expected MAJOR.MINOR.PATCH "
            f"(optional suffix like -rc1), e.g. 1.2.34"
        )


def read_current_version(root: Path = ROOT) -> str:
    version_py = (root / "src" / "version.py").read_text(encoding="utf-8")
    match = APP_VERSION_RE.search(version_py)
    if not match:
        raise ValueError("Could not read APP_VERSION from src/version.py")
    return match.group(1)


def extract_version(rel_path: str, content: str) -> str:
    if rel_path == "src/version.py":
        match = APP_VERSION_RE.search(content)
    else:
        match = MY_APP_VERSION_RE.search(content)
    if not match:
        raise ValueError(f"Could not find version marker in {rel_path}")
    return match.group(1)


def apply_version(content: str, rel_path: str, new_version: str) -> str:
    if rel_path == "src/version.py":
        updated, count = APP_VERSION_RE.subn(
            f'APP_VERSION = "{new_version}"',
            content,
            count=1,
        )
    else:
        updated, count = MY_APP_VERSION_RE.subn(
            f'#define MyAppVersion "{new_version}"',
            content,
            count=1,
        )
    if count != 1:
        raise ValueError(f"Failed to update version in {rel_path}")
    return updated


def bump_version_files(new_version: str, root: Path = ROOT) -> list[FileChange]:
    validate_version_string(new_version)

    pending_writes: list[tuple[str, str, str, str]] = []

    for rel_path in VERSION_FILES:
        path = root / rel_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing version file: {rel_path}")

        content = path.read_text(encoding="utf-8")
        old_version = extract_version(rel_path, content)
        updated = apply_version(content, rel_path, new_version)
        pending_writes.append((rel_path, content, updated, old_version))

    write_results: list[tuple[str, bool, str | None]] = []
    changes: list[FileChange] = []

    for rel_path, _original, updated, old_version in pending_writes:
        path = root / rel_path
        try:
            path.write_text(updated, encoding="utf-8")
            write_results.append((rel_path, True, None))
            changes.append(
                FileChange(rel_path=rel_path, old_version=old_version, new_version=new_version)
            )
        except OSError as exc:
            write_results.append((rel_path, False, str(exc)))

    failures = [rel for rel, ok, _ in write_results if not ok]
    if failures:
        print("ERROR: Version bump incomplete. Write results:", file=sys.stderr)
        for rel_path, ok, err in write_results:
            status = "OK" if ok else f"FAILED ({err})"
            print(f"  {rel_path}: {status}", file=sys.stderr)
        raise RuntimeError(f"Failed to write: {', '.join(failures)}")

    return changes


def run_validate_ci_build(root: Path = ROOT) -> int:
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "validate_ci_build.py")],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    return result.returncode


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Bump APP_VERSION in src/version.py and matching MyAppVersion "
            "defines in packaging/*.iss, then run validate_ci_build.py."
        ),
    )
    parser.add_argument(
        "version",
        help="New version (MAJOR.MINOR.PATCH, optional suffix like -rc1)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    effective_root = root if root is not None else args.root

    try:
        validate_version_string(args.version)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        old_version = read_current_version(effective_root)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    new_version = args.version
    print(f"{old_version} -> {new_version}")

    if old_version == new_version:
        print("Version unchanged; running validation only.")

    try:
        changes = bump_version_files(new_version, root=effective_root)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nUpdated files:")
    for change in changes:
        print(f"  {change.rel_path}: {change.old_version} -> {change.new_version}")

    print("\nRunning scripts/validate_ci_build.py ...")
    exit_code = run_validate_ci_build(root=effective_root)
    if exit_code != 0:
        print("ERROR: validate_ci_build.py failed after bump.", file=sys.stderr)
        return exit_code

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
