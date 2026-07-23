#!/usr/bin/env bash
# Build TileVision AI for macOS — Intel (x64) and/or Apple Silicon (arm64).
#
# Usage:
#   bash scripts/build_mac.sh              # native arch (arm64 on M Mac, x64 on Intel Mac)
#   MACOS_ARCH=x64 bash scripts/build_mac.sh      # Intel Mac customers
#   MACOS_ARCH=arm64 bash scripts/build_mac.sh     # Apple Silicon customers
#   MACOS_ARCH=both bash scripts/build_mac.sh      # both DMGs (full release)
#
# Matches GitHub Actions build-macos matrix (same venv + verify scripts).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -n "${TILEVISION_DEV_MODE:-}" ]]; then
  echo "ERROR: TILEVISION_DEV_MODE is set — unset it before a production build."
  exit 1
fi

build_one() {
  local arch="$1"
  local label dmg verify
  case "$arch" in
    x64)
      label="Intel"
      dmg="TileVisionAI-macOS-Intel.dmg"
      verify="x86_64"
      ;;
    arm64)
      label="Apple Silicon"
      dmg="TileVisionAI-macOS-AppleSilicon.dmg"
      verify="arm64"
      ;;
    *)
      echo "ERROR: unknown arch $arch" >&2
      exit 1
      ;;
  esac

  echo
  echo "========== Building macOS $label ($verify) =========="

  # Local: use python3 from PATH. CI sets PYTHON_SETUP_PATH before calling this script.
  export PYTHON_SETUP_PATH="${PYTHON_SETUP_PATH:-$(command -v python3)}"
  # shellcheck disable=SC1091
  source scripts/install_mac_deps.sh "$arch"
  export MACOS_BUILD_ARCH="$arch"

  echo "[1/5] Dependencies installed (install_mac_deps.sh)"

  echo "[2/5] Verifying native library architecture..."
  bash scripts/verify_mac_native_libs.sh "$verify"

  echo "[3/5] Ensuring DINOv2 model weights..."
  MODEL_DIR="model_weights/dinov2-large"
  if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    echo "  Downloading DINOv2 (~1 GB)..."
    bash scripts/macos_build_python.sh scripts/download_dinov2_model.py
  else
    echo "  Model weights already present at $MODEL_DIR"
  fi

  export TILEVISION_OFFLINE_MODEL=1

  echo "[4/5] Running PyInstaller..."
  rm -rf build dist/TileVisionAI dist/TileVisionAI.app
  bash scripts/macos_build_python.sh -m PyInstaller packaging/tilevision_mac.spec --clean --noconfirm

  echo "[5/5] Verifying frozen app + creating DMG..."
  bash scripts/verify_frozen_mac_app.sh dist/TileVisionAI.app "$verify"

  if command -v hdiutil >/dev/null 2>&1; then
    rm -f "dist/$dmg"
    hdiutil create -volname "TileVision AI ($label)" -srcfolder dist/TileVisionAI.app \
      -ov -format UDZO "dist/$dmg"
    echo "  DMG: dist/$dmg"
  fi

  echo "Done: dist/TileVisionAI.app ($label)"
}

HOST_ARCH="$(uname -m)"
TARGET="${MACOS_ARCH:-}"
if [[ -z "$TARGET" ]]; then
  if [[ "$HOST_ARCH" == "arm64" ]]; then
    TARGET="arm64"
  else
    TARGET="x64"
  fi
fi

echo "=== TileVision AI — macOS Release Build (target: $TARGET) ==="

case "$TARGET" in
  both)
    build_one x64
    build_one arm64
    if [[ -f "dist/TileVisionAI-macOS-Intel.dmg" && -f "dist/TileVisionAI-macOS-AppleSilicon.dmg" ]]; then
      bash scripts/package_mac_universal.sh \
        "dist/TileVisionAI-macOS-Intel.dmg" \
        "dist/TileVisionAI-macOS-AppleSilicon.dmg" \
        "dist/TileVisionAI-macOS-local.zip"
      echo "Universal zip: dist/TileVisionAI-macOS-local.zip"
    fi
    ;;
  x64|arm64)
    build_one "$TARGET"
    ;;
  *)
    echo "ERROR: MACOS_ARCH must be x64, arm64, or both (got: $TARGET)" >&2
    exit 1
    ;;
esac

echo
echo "Ship the .dmg (or universal zip) to Mac customers."
echo "First launch: Right-click → Open (unsigned build)."
