#!/usr/bin/env bash
# Build TileVision AI for macOS: PyInstaller + optional DMG.
#
# Run on a Mac (Apple Silicon or Intel):
#   bash scripts/build_mac.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== TileVision AI — macOS Release Build ==="

if [[ -n "${TILEVISION_DEV_MODE:-}" ]]; then
  echo "ERROR: TILEVISION_DEV_MODE is set — unset it before a production build."
  exit 1
fi

# Intel x64 runs on Intel Macs. On Apple Silicon build hosts use an isolated x86_64 venv.
if [[ "$(uname -m)" == "arm64" ]]; then
  echo "Apple Silicon detected — building x86_64 in Rosetta venv for Intel Mac showrooms."
  PY="$(command -v python3)"
  VENV=".venv-macos-x64"
  arch -x86_64 "$PY" -m venv "$VENV"
  PYTHON="arch -x86_64 $VENV/bin/python"
else
  PYTHON="python3"
fi

echo
echo "[1/4] Checking Python dependencies..."
$PYTHON -m pip install --upgrade pip >/dev/null
$PYTHON -m pip install --no-cache-dir -r requirements.txt pyinstaller

echo
echo "[2/4] Ensuring DINOv2 model weights..."
MODEL_DIR="model_weights/dinov2-large"
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "  Downloading DINOv2 (~1 GB)..."
  $PYTHON scripts/download_dinov2_model.py
else
  echo "  Model weights already present at $MODEL_DIR"
fi

export TILEVISION_OFFLINE_MODEL=1

echo
echo "[3/4] Running PyInstaller..."
rm -rf build dist/TileVisionAI dist/TileVisionAI.app
$PYTHON -m PyInstaller packaging/tilevision_mac.spec --clean --noconfirm

APP_PATH="dist/TileVisionAI.app"
if [[ ! -d "$APP_PATH" ]]; then
  echo "ERROR: PyInstaller build failed — $APP_PATH not found."
  exit 1
fi
echo "  Built: $APP_PATH"

echo
echo "[4/4] Creating DMG (optional)..."
DMG_PATH="dist/TileVisionAI-macOS.dmg"
if command -v hdiutil >/dev/null 2>&1; then
  rm -f "$DMG_PATH"
  hdiutil create -volname "TileVision AI" -srcfolder "$APP_PATH" -ov -format UDZO "$DMG_PATH"
  echo "  DMG: $DMG_PATH"
else
  echo "  hdiutil not found — zip the .app manually."
fi

echo
echo "Done."
echo "  App:  $APP_PATH"
echo "  Ship the .app or .dmg to Mac customers."
echo "  First launch: Right-click → Open (unsigned build)."
