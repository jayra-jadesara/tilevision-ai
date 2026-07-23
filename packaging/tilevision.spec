# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for TileVision AI (Windows)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(SPECPATH)))
from pyinstaller_common import EXCLUDES, HIDDEN_IMPORTS, collect_datas

block_cipher = None
PROJECT_ROOT = Path(SPECPATH).parent

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=collect_datas(PROJECT_ROOT),
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
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
    upx=True,
    console=False,
    icon=str(PROJECT_ROOT / "src" / "resources" / "app_icon.ico")
    if (PROJECT_ROOT / "src" / "resources" / "app_icon.ico").exists()
    else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TileVisionAI",
)
