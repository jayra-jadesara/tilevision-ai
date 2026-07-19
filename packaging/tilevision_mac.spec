# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build spec for TileVision AI (macOS).

Build on a Mac (Apple Silicon or Intel):

    pip install -r requirements.txt pyinstaller
    python scripts/download_dinov2_model.py
    pyinstaller packaging/tilevision_mac.spec --clean

Output: dist/TileVisionAI.app
"""

import sys
from pathlib import Path

block_cipher = None

PROJECT_ROOT = Path(SPECPATH).parent
MODEL_DIR = PROJECT_ROOT / "model_weights" / "dinov2-large"

datas = []
if MODEL_DIR.is_dir():
    datas.append((str(MODEL_DIR), str(Path("model_weights") / "dinov2-large")))

resources = PROJECT_ROOT / "src" / "resources"
if resources.is_dir():
    datas.append((str(resources), "src/resources"))

hidden_imports = [
    "transformers",
    "transformers.models.dinov2",
    "timm",
    "safetensors",
    "tokenizers",
    "huggingface_hub",
    "torch",
    "torchvision",
    "faiss",
    "cv2",
    "PIL",
    "skimage",
    "cryptography",
    "watchdog.observers",
    "watchdog.events",
]

a = Analysis(
    [str(PROJECT_ROOT / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "notebook", "jupyter"],
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
    icon=str(PROJECT_ROOT / "src" / "resources" / "app_icon.png")
    if (PROJECT_ROOT / "src" / "resources" / "app_icon.png").exists()
    else None,
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
    name="TileVisionAI.app",
    icon=str(PROJECT_ROOT / "src" / "resources" / "app_icon.png")
    if (PROJECT_ROOT / "src" / "resources" / "app_icon.png").exists()
    else None,
    bundle_identifier="com.jdsoftware.tilevisionai",
)
