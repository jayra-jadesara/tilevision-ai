#!/usr/bin/env bash
# Post-PyInstaller checks for TileVisionAI.app — Intel and Apple Silicon.

set -euo pipefail

APP="${1:?path to TileVisionAI.app}"
EXPECTED="${2:?expected arch: x86_64 or arm64}"

MACOS_DIR="$APP/Contents/MacOS"
BIN="$MACOS_DIR/TileVisionAI"

echo "=== Verifying frozen Mac app ($EXPECTED) ==="

if [[ ! -f "$BIN" ]]; then
  echo "ERROR: missing executable: $BIN" >&2
  exit 1
fi

file "$BIN"
file "$BIN" | grep -qE "${EXPECTED}|universal"

MODEL="$(find "$APP" -path '*/model_weights/dinov2-large/config.json' 2>/dev/null | head -n 1 || true)"
if [[ -z "$MODEL" ]]; then
  echo "ERROR: DINOv2 model not bundled in .app" >&2
  exit 1
fi
echo "model bundled: $MODEL"

# Optional SAM2 Precise Crop — ONNX required for Mac Intel + Silicon when bundling.
BUNDLE_SAM2="${TILEVISION_BUNDLE_SAM2:-}"
BUNDLE_SAM2_LC="$(printf '%s' "$BUNDLE_SAM2" | tr '[:upper:]' '[:lower:]')"
expect_sam2=0
case "$BUNDLE_SAM2_LC" in
  1|true|yes|on|auto) expect_sam2=1 ;;
esac
if [[ "$expect_sam2" == "1" ]]; then
  SAM2_ONNX="$(find "$APP" -path '*/model_weights/sam2.1-hiera-tiny-onnx/*.encoder.onnx' 2>/dev/null | head -n 1 || true)"
  if [[ -z "$SAM2_ONNX" ]]; then
    echo "ERROR: TILEVISION_BUNDLE_SAM2=$BUNDLE_SAM2 but ONNX SAM2 encoder not in .app" >&2
    exit 1
  fi
  echo "sam2 onnx bundled: $SAM2_ONNX"
  if [[ "$EXPECTED" == "arm64" ]]; then
    SAM2_TR="$(find "$APP" -path '*/model_weights/sam2.1-hiera-tiny/config.json' 2>/dev/null | head -n 1 || true)"
    if [[ -n "$SAM2_TR" ]]; then
      echo "sam2 transformers bundled: $SAM2_TR"
    else
      echo "sam2 transformers optional missing (ONNX still present)"
    fi
  else
    echo "Mac Intel: ONNX SAM2 path verified (Transformers not required)"
  fi
else
  echo "sam2 bundle skipped (TILEVISION_BUNDLE_SAM2=${BUNDLE_SAM2:-off})"
fi

TORCH="$(find "$APP" -type d -path '*/torch' 2>/dev/null | head -n 1 || true)"
if [[ -z "$TORCH" ]]; then
  echo "ERROR: torch package missing from .app bundle" >&2
  exit 1
fi
echo "torch: $TORCH"

if [[ ! -d "$TORCH/cuda" ]]; then
  echo "ERROR: torch.cuda missing from .app bundle — app will crash on startup" >&2
  exit 1
fi
echo "torch.cuda: $TORCH/cuda"

echo "=== Frozen Mac app OK ($EXPECTED) ==="
