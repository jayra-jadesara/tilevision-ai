#!/usr/bin/env bash
# Build TileVision AI for macOS — Intel (x64) and/or Apple Silicon (arm64).
#
# Universal2 is NOT supported: Intel pins torch==2.2.2; Apple Silicon uses
# current torch from requirements.txt. Those ABIs cannot share one fat .app.
#
# Outputs (customer-facing names):
#   dist/TileVision-AI-Intel.dmg
#   dist/TileVision-AI-AppleSilicon.dmg
#
# Legacy aliases (update-check / older docs):
#   dist/TileVisionAI-macOS-Intel.dmg
#   dist/TileVisionAI-macOS-AppleSilicon.dmg
#
# Usage:
#   bash scripts/build_mac.sh
#   MACOS_ARCH=x64|arm64|both bash scripts/build_mac.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -n "${TILEVISION_DEV_MODE:-}" ]]; then
  echo "ERROR: TILEVISION_DEV_MODE is set — unset it before a production build."
  exit 1
fi

APP_BUNDLE='dist/TileVision AI.app'

build_one() {
  local arch="$1"
  local label dmg legacy verify
  case "$arch" in
    x64)
      label="Intel"
      dmg="TileVision-AI-Intel.dmg"
      legacy="TileVisionAI-macOS-Intel.dmg"
      verify="x86_64"
      ;;
    arm64)
      label="Apple Silicon"
      dmg="TileVision-AI-AppleSilicon.dmg"
      legacy="TileVisionAI-macOS-AppleSilicon.dmg"
      verify="arm64"
      ;;
    *)
      echo "ERROR: unknown arch $arch" >&2
      exit 1
      ;;
  esac

  echo
  echo "========== Building macOS $label ($verify) =========="

  export PYTHON_SETUP_PATH="${PYTHON_SETUP_PATH:-$(command -v python3)}"
  # shellcheck disable=SC1091
  source scripts/install_mac_deps.sh "$arch"
  export MACOS_BUILD_ARCH="$arch"

  echo "[1/6] Dependencies installed (install_mac_deps.sh)"

  echo "[2/6] Verifying native library architecture..."
  bash scripts/verify_mac_native_libs.sh "$verify"

  echo "[3/6] Ensuring DINOv2 model weights..."
  MODEL_DIR="model_weights/dinov2-large"
  if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    echo "  Downloading DINOv2 (~1 GB)..."
    bash scripts/macos_build_python.sh scripts/download_dinov2_model.py
  else
    echo "  Model weights already present at $MODEL_DIR"
  fi

  export TILEVISION_BUNDLE_SAM2="${TILEVISION_BUNDLE_SAM2:-auto}"
  if [[ "$TILEVISION_BUNDLE_SAM2" != "0" && "$TILEVISION_BUNDLE_SAM2" != "off" && "$TILEVISION_BUNDLE_SAM2" != "false" ]]; then
    bundle_any=0
    case "$TILEVISION_BUNDLE_SAM2" in
      1|true|yes|on|auto) bundle_any=1 ;;
    esac
    if [[ "$bundle_any" == "1" ]]; then
      echo "[3b/6] Ensuring ONNX SAM2 Precise Crop weights..."
      ONNX_DIR="model_weights/sam2.1-hiera-tiny-onnx"
      if [[ ! -f "$ONNX_DIR/sam2.1_hiera_tiny.encoder.onnx" && ! -f "$ONNX_DIR/encoder.onnx" ]]; then
        echo "  Downloading ONNX SAM2 tiny (~126 MB)..."
        bash scripts/macos_build_python.sh scripts/download_sam2_onnx_model.py
      else
        echo "  ONNX SAM2 weights already present at $ONNX_DIR"
      fi
      TR_FLAG="$(printf '%s' "${TILEVISION_BUNDLE_SAM2_TRANSFORMERS:-}" | tr '[:upper:]' '[:lower:]')"
      if [[ "$TR_FLAG" == "1" || "$TR_FLAG" == "true" || "$TR_FLAG" == "yes" || "$TR_FLAG" == "on" ]]; then
        echo "[3c/6] Ensuring optional Transformers SAM2 weights..."
        SAM2_DIR="model_weights/sam2.1-hiera-tiny"
        if [[ ! -f "$SAM2_DIR/config.json" ]]; then
          bash scripts/macos_build_python.sh scripts/download_sam2_model.py
        else
          echo "  Transformers SAM2 weights already present at $SAM2_DIR"
        fi
      else
        echo "[3c/6] Transformers SAM2 skipped (ONNX is the shared Mac/Windows path)."
      fi
    fi
  fi

  export TILEVISION_OFFLINE_MODEL=1

  echo "[4/6] Running PyInstaller..."
  rm -rf build dist/TileVisionAI "$APP_BUNDLE"
  bash scripts/macos_build_python.sh -m PyInstaller packaging/tilevision_mac.spec --clean --noconfirm

  echo "[5/6] Verifying frozen app..."
  bash scripts/verify_frozen_mac_app.sh "$APP_BUNDLE" "$verify"

  if [[ "${TILEVISION_SKIP_SMOKE:-0}" != "1" ]]; then
    echo "[5b/6] Smoke-launching frozen app..."
    bash scripts/smoke_launch_mac_app.sh "$APP_BUNDLE" || {
      echo "ERROR: smoke launch failed" >&2
      exit 1
    }
  else
    echo "[5b/6] Smoke launch skipped (TILEVISION_SKIP_SMOKE=1)"
  fi

  echo "[6/6] Creating professional DMG..."
  if command -v hdiutil >/dev/null 2>&1; then
    rm -f "dist/$dmg" "dist/$legacy"
    bash scripts/create_mac_dmg.sh "$APP_BUNDLE" "dist/$dmg" "TileVision AI"
    # Keep legacy filename for update-check / older docs.
    cp "dist/$dmg" "dist/$legacy"
    echo "  DMG: dist/$dmg"
    echo "  Legacy alias: dist/$legacy"
  fi

  echo "[report] Writing packaging reports..."
  bash scripts/macos_build_python.sh scripts/generate_macos_packaging_report.py \
    --app "$APP_BUNDLE" \
    --dmg "dist/$dmg" \
    --arch "$verify" \
    --out-dir "dist/packaging_reports/$verify"

  echo "Done: $APP_BUNDLE ($label)"
  echo "SIGNING: Unsigned build (no Apple Developer credentials in this pipeline)."
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
echo "Universal2: NOT SUPPORTED (see packaging/MACOS_RELEASE.md)"

case "$TARGET" in
  both)
    build_one x64
    build_one arm64
    if [[ -f "dist/TileVision-AI-Intel.dmg" && -f "dist/TileVision-AI-AppleSilicon.dmg" ]]; then
      bash scripts/package_mac_universal.sh \
        "dist/TileVision-AI-Intel.dmg" \
        "dist/TileVision-AI-AppleSilicon.dmg" \
        "dist/TileVision-AI-macOS-both.zip"
      echo "Dual-arch zip (NOT Universal2): dist/TileVision-AI-macOS-both.zip"
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
echo "Ship the .dmg to Mac customers (correct arch)."
echo "First launch: Right-click the app → Open (Unsigned build)."
