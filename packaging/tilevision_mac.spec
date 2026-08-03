# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for TileVision AI (macOS).

Produces a one-folder .app with DINOv2 (+ optional SAM2 ONNX) bundled.
Customer machines do not need Python installed.

Universal2 is NOT produced here — Intel requires torch 2.2.2 wheels while
Apple Silicon uses current torch; those cannot be lipo'd into one binary.
Build Intel and Apple Silicon DMGs separately.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(SPECPATH)))
from pyinstaller_common import (
    EXCLUDES,
    collect_datas,
    collect_hidden_imports,
    collect_torch_bundle,
    collect_extra_package_bundles,
)

block_cipher = None
PROJECT_ROOT = Path(SPECPATH).parent

# Import version without importing the full app.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from version import APP_VERSION  # noqa: E402

_torch_datas, _torch_binaries, _torch_hidden = collect_torch_bundle()
_extra_datas, _extra_binaries, _extra_hidden = collect_extra_package_bundles()

_icon = PROJECT_ROOT / "src" / "resources" / "app_icon.png"
_icon_path = str(_icon) if _icon.exists() else None

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=_torch_binaries + _extra_binaries,
    datas=collect_datas(PROJECT_ROOT) + _torch_datas + _extra_datas,
    hiddenimports=collect_hidden_imports() + _torch_hidden + _extra_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    cipher=block_cipher,
    noarchive=False,
    # Keep cv2 as source files so its macOS bootstrap can locate the extension
    # beside the package (avoids "recursion is detected during loading of cv2").
    module_collection_mode={"cv2": "py"},
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TileVisionAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=_icon_path,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TileVisionAI",
)

app = BUNDLE(
    coll,
    name="TileVision AI.app",
    icon=_icon_path,
    bundle_identifier="com.jdsoftware.tilevisionai",
    info_plist={
        "CFBundleName": "TileVision AI",
        "CFBundleDisplayName": "TileVision AI",
        "CFBundleIdentifier": "com.jdsoftware.tilevisionai",
        "CFBundleVersion": APP_VERSION,
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundlePackageType": "APPL",
        "CFBundleExecutable": "TileVisionAI",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,
        "NSHumanReadableCopyright": "Copyright © JD Software. All rights reserved.",
        "LSApplicationCategoryType": "public.app-category.graphics-design",
    },
)
