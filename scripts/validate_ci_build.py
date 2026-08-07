#!/usr/bin/env python3
"""Fast pre-build checks — run in every CI build job before PyInstaller."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KNOWN_VERSION_FILES = frozenset(
    {
        "src/version.py",
        "packaging/tilevision_setup.iss",
        "packaging/tilevision_admin_setup.iss",
    }
)

VERSION_DEF_LINE = re.compile(
    r'(#define\s+MyAppVersion\s+"(\d+\.\d+\.\d+[^"]*)"|'
    r'APP_VERSION\s*=\s*"(\d+\.\d+\.\d+[^"]*)"|'
    r'version\s*=\s*"(\d+\.\d+\.\d+[^"]*)")',
    re.IGNORECASE,
)


def main() -> int:
    errors: list[str] = []

    for rel in (
        "packaging/pyinstaller_common.py",
        "packaging/tilevision.spec",
        "packaging/tilevision_mac.spec",
        "packaging/tilevision_linux.spec",
        "packaging/tilevision_admin.spec",
        "packaging/tilevision_admin_setup.iss",
        "packaging/VENDOR_ADMIN_README.txt",
        "packaging/tilevision_setup.iss",
        "packaging/MAC_INSTALL.txt",
        "scripts/install_mac_deps.sh",
        "scripts/macos_build_python.sh",
        "scripts/smoke_test_windows.ps1",
        "scripts/run_pre_release_tests.ps1",
        "packaging/MAC_BETA_TEST.txt",
        "scripts/verify_mac_native_libs.sh",
        "scripts/verify_frozen_mac_app.sh",
        "scripts/create_mac_dmg.sh",
        "scripts/free_mac_runner_disk.sh",
        "scripts/verify_frozen_windows.ps1",
        "scripts/package_mac_universal.sh",
        "src/version.py",
    ):
        if not (ROOT / rel).is_file():
            errors.append(f"missing file: {rel}")

    sys.path.insert(0, str(ROOT / "packaging"))
    try:
        from pyinstaller_common import HIDDEN_IMPORTS, collect_datas  # noqa: PLC0415

        if len(HIDDEN_IMPORTS) < 10:
            errors.append("HIDDEN_IMPORTS looks too small")
        collect_datas(ROOT)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"pyinstaller_common import failed: {exc}")

    version_py = (ROOT / "src" / "version.py").read_text(encoding="utf-8")
    app_version = next(
        (line.split('"')[1] for line in version_py.splitlines() if line.startswith("APP_VERSION")),
        "",
    )
    iss = (ROOT / "packaging" / "tilevision_setup.iss").read_text(encoding="utf-8")
    for line in iss.splitlines():
        if "#define MyAppVersion" in line:
            iss_version = line.split('"')[1]
            break
    else:
        iss_version = ""

    admin_iss = (ROOT / "packaging" / "tilevision_admin_setup.iss").read_text(encoding="utf-8")
    for line in admin_iss.splitlines():
        if "#define MyAppVersion" in line:
            admin_iss_version = line.split('"')[1]
            break
    else:
        admin_iss_version = ""

    if app_version != iss_version:
        errors.append(f"version mismatch: version.py={app_version} iss={iss_version}")
    if app_version != admin_iss_version:
        errors.append(
            f"version mismatch: version.py={app_version} admin_iss={admin_iss_version}"
        )

    scan_paths: list[Path] = [ROOT / "src" / "version.py"]
    packaging_dir = ROOT / "packaging"
    if packaging_dir.is_dir():
        scan_paths.extend(sorted(packaging_dir.rglob("*")))

    for path in scan_paths:
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in KNOWN_VERSION_FILES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if VERSION_DEF_LINE.search(line):
                print(
                    f"WARNING: possible hardcoded version in {rel}:{line_no} "
                    f"(not in known bump list — add to scripts/bump_version.py if intentional)",
                    file=sys.stderr,
                )
                break

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print(f"CI build config OK (version {app_version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
