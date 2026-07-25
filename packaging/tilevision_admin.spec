# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec — Vendor Admin License Manager (Windows only, do not ship to customers)."""

from pathlib import Path

block_cipher = None
PROJECT_ROOT = Path(SPECPATH).parent
ADMIN_DIR = PROJECT_ROOT / "admin_tool"

ADMIN_HIDDEN = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "cryptography",
    "cryptography.hazmat.primitives.asymmetric.ec",
    "cryptography.hazmat.primitives.serialization",
    "cryptography.hazmat.primitives.hashes",
    "cryptography.exceptions",
    "sqlite3",
    "license_ledger",
    "admin_theme",
    "vendor_backup",
    "web_date_picker",
    "src.licensing.validator",
    "src.licensing.hardware",
    "src.licensing.revocation",
    "src.utils.brand_assets",
    "src.utils.platform_info",
    "src.theme.theme_manager",
]

ADMIN_DATAS: list[tuple[str, str]] = []
resources = PROJECT_ROOT / "src" / "resources"
if resources.is_dir():
    ADMIN_DATAS.append((str(resources), "src/resources"))

ADMIN_EXCLUDES = [
    "torch",
    "torchvision",
    "transformers",
    "timm",
    "faiss",
    "cv2",
    "PIL",
    "matplotlib",
    "notebook",
    "jupyter",
]

a = Analysis(
    [str(ADMIN_DIR / "main.py")],
    pathex=[str(PROJECT_ROOT), str(ADMIN_DIR)],
    binaries=[],
    datas=ADMIN_DATAS,
    hiddenimports=ADMIN_HIDDEN,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=ADMIN_EXCLUDES,
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
    name="TileVisionAI-Admin",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
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
    upx=False,
    upx_exclude=[],
    name="TileVisionAI-Admin",
)
