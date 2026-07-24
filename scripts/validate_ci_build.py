#!/usr/bin/env python3
"""Fast pre-build checks — run in every CI build job before PyInstaller."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    errors: list[str] = []

    for rel in (
        "packaging/pyinstaller_common.py",
        "packaging/tilevision.spec",
        "packaging/tilevision_mac.spec",
        "packaging/tilevision_linux.spec",
        "packaging/tilevision_setup.iss",
        "packaging/MAC_INSTALL.txt",
        "scripts/install_mac_deps.sh",
        "scripts/macos_build_python.sh",
        "scripts/smoke_test_windows.ps1",
        "scripts/run_pre_release_tests.ps1",
        "packaging/MAC_BETA_TEST.txt",
        "scripts/verify_mac_native_libs.sh",
        "scripts/verify_frozen_mac_app.sh",
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

    if app_version != iss_version:
        errors.append(f"version mismatch: version.py={app_version} iss={iss_version}")

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print(f"CI build config OK (version {app_version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
