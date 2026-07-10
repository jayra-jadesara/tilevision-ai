# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build spec for TileVision AI.

Build with (from the project root, in an activated venv with requirements.txt
installed AND PyInstaller installed):

    pyinstaller packaging/tilevision.spec --clean

Output: dist/TileVisionAI/TileVisionAI.exe (a one-folder build — recommended
over --onefile for this app, since --onefile re-extracts the ~1-2GB of
torch/CLIP model weights to a temp folder on every launch, adding many
seconds to every startup).

IMPORTANT — Offline model weights:
    By default, open_clip's `pretrained="laion400m_e32"` downloads weights
    from a remote hub the first time the app runs, which violates the "no
    internet dependency after installation" requirement. Before building a
    release, you must pre-download the weights once (with internet access,
    on a build machine) and either:
      (a) point src/config/settings.py's default `pretrained` value at a
          local .bin/.pt weights file path instead of the dataset name, or
      (b) let open_clip cache the weights to its default cache dir once,
          then add that cache directory to the `datas` list below so
          PyInstaller bundles it into the installer.
    Without one of these, first launch on a customer's offline machine will
    fail to load the AI model.
"""

import sys
from pathlib import Path

block_cipher = None

PROJECT_ROOT = Path(SPECPATH).parent  # packaging/ -> project root

# ── Data files bundled into the executable ──────────────────────────────
# Add the pre-downloaded CLIP weights directory here once you have one, e.g.:
#   (str(PROJECT_ROOT / "model_weights"), "model_weights"),
datas = [
    (str(PROJECT_ROOT / "src" / "config" / "default_config.json"), "src/config"),
] if (PROJECT_ROOT / "src" / "config" / "default_config.json").exists() else []

# ── Hidden imports ───────────────────────────────────────────────────────
# PyInstaller's static analysis frequently misses dynamically-loaded pieces
# of torch/open_clip/faiss; list them explicitly to avoid a runtime
# ModuleNotFoundError deep inside the AI pipeline on the customer's machine.
hidden_imports = [
    "open_clip",
    "torch",
    "torchvision",
    "faiss",
    "cv2",
    "PIL",
    "cryptography",
    "cryptography.hazmat.primitives.ciphers.aead",
    "cryptography.hazmat.primitives.kdf.pbkdf2",
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
    excludes=[
        # Trim obviously unused heavy extras to reduce installer size.
        "matplotlib",
        "notebook",
        "jupyter",
    ],
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
    console=False,  # windowed app — no console window on launch
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
