#!/usr/bin/env bash
# Post-PyInstaller checks for TileVision AI.app — Intel and Apple Silicon.

set -euo pipefail

APP="${1:?path to TileVision AI.app}"
EXPECTED="${2:?expected arch: x86_64 or arm64}"

MACOS_DIR="$APP/Contents/MacOS"
BIN="$MACOS_DIR/TileVisionAI"
PLIST="$APP/Contents/Info.plist"

echo "=== Verifying frozen Mac app ($EXPECTED) ==="
echo "app: $APP"

if [[ ! -d "$APP" ]]; then
  echo "ERROR: app bundle missing: $APP" >&2
  exit 1
fi

if [[ ! -f "$BIN" ]]; then
  echo "ERROR: missing executable: $BIN" >&2
  exit 1
fi

file "$BIN"
file "$BIN" | grep -qE "${EXPECTED}|universal"

if [[ -f "$PLIST" ]]; then
  echo "Info.plist present"
  /usr/libexec/PlistBuddy -c 'Print CFBundleIdentifier' "$PLIST" 2>/dev/null || true
  /usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "$PLIST" 2>/dev/null || true
  /usr/libexec/PlistBuddy -c 'Print CFBundleDisplayName' "$PLIST" 2>/dev/null || true
else
  echo "WARNING: Info.plist missing" >&2
fi

MODEL="$(find "$APP" -path '*/model_weights/dinov2-large/config.json' 2>/dev/null | head -n 1 || true)"
if [[ -z "$MODEL" ]]; then
  echo "ERROR: DINOv2 model not bundled in .app" >&2
  exit 1
fi
echo "model bundled: $MODEL"

# Optional SAM2 Precise Crop — ONNX required when bundling.
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

# Critical runtime packages / Qt plugins
for needle in faiss cv2 onnxruntime reportlab PySide6; do
  hit="$(find "$APP" -iname "*${needle}*" 2>/dev/null | head -n 1 || true)"
  if [[ -z "$hit" ]]; then
    echo "ERROR: expected package content for $needle not found in .app" >&2
    exit 1
  fi
  echo "found $needle: $hit"
done

# Qt platform plugin (offscreen/cocoa)
QT_PLUGINS="$(find "$APP" -type d -name 'platforms' 2>/dev/null | head -n 1 || true)"
if [[ -z "$QT_PLUGINS" ]]; then
  echo "WARNING: Qt platforms plugin directory not found (may still resolve via PySide6)" >&2
else
  echo "Qt platforms: $QT_PLUGINS"
  ls "$QT_PLUGINS" | head -20
fi

# Resources / icons
RES="$(find "$APP" -path '*/src/resources/app_icon.png' 2>/dev/null | head -n 1 || true)"
if [[ -z "$RES" ]]; then
  echo "WARNING: app_icon.png not found under src/resources in bundle" >&2
else
  echo "icon resource: $RES"
fi

echo "=== Frozen Mac app OK ($EXPECTED) ==="
