#!/usr/bin/env bash
# Post-PyInstaller checks for TileVisionAI.app — Intel and Apple Silicon.
#
# Usage:
#   scripts/verify_frozen_mac_app.sh dist/TileVisionAI.app x86_64
#   scripts/verify_frozen_mac_app.sh dist/TileVisionAI.app arm64

set -euo pipefail

APP="${1:?path to TileVisionAI.app}"
EXPECTED="${2:?expected arch: x86_64 or arm64}"

MACOS_DIR="$APP/Contents/MacOS"
BIN="$MACOS_DIR/TileVisionAI"

echo "=== Verifying frozen Mac app ($EXPECTED) ==="

if [[ ! -f "$BIN" ]]; then
  echo "ERROR: missing executable: $BIN" >&2
  find "$APP" -maxdepth 4 -type f 2>/dev/null | head -20
  exit 1
fi

file "$BIN"
file "$BIN" | grep -qE "${EXPECTED}|universal"

MODEL="$(find "$APP" -path '*/model_weights/dinov2-large/config.json' 2>/dev/null | head -n 1 || true)"
if [[ -z "$MODEL" ]]; then
  echo "ERROR: DINOv2 model not bundled in .app" >&2
  find "$APP" -maxdepth 5 -type d 2>/dev/null | head -30
  exit 1
fi
echo "model bundled: $MODEL"

echo "=== Frozen Mac app OK ($EXPECTED) ==="
