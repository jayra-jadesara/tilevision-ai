# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for TileVision AI (Linux)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(SPECPATH)))
from pyinstaller_common import (
    EXCLUDES,
    collect_datas,
    collect_hidden_imports,
    collect_torch_bundle,
)

block_cipher = None
PROJECT_ROOT = Path(SPECPATH).parent

_torch_datas, _torch_binaries, _torch_hidden = collect_torch_bundle()

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=_torch_binaries,
    datas=collect_datas(PROJECT_ROOT) + _torch_datas,
    hiddenimports=collect_hidden_imports() + _torch_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
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
    upx=False,
    console=False,
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
