#!/usr/bin/env bash
# Post-PyInstaller checks for TileVisionAI.app — same rules for Intel and Apple Silicon.
#
# Usage:
#   scripts/verify_frozen_mac_app.sh dist/TileVisionAI.app x86_64
#   scripts/verify_frozen_mac_app.sh dist/TileVisionAI.app arm64

set -euo pipefail

APP="${1:?path to TileVisionAI.app}"
EXPECTED="${2:?expected arch: x86_64 or arm64}"

BIN="$APP/Contents/MacOS/TileVisionAI"
MODEL="$APP/Contents/MacOS/model_weights/dinov2-large/config.json"
MACOS_DIR="$APP/Contents/MacOS"

echo "=== Verifying frozen Mac app ($EXPECTED) ==="

if [[ ! -f "$BIN" ]]; then
  echo "ERROR: missing executable: $BIN" >&2
  exit 1
fi

file "$BIN"
file "$BIN" | grep -q "$EXPECTED"

if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: offline model not bundled: $MODEL" >&2
  exit 1
fi
echo "model bundled: $MODEL"

check_bundle_lib() {
  local pattern="$1"
  local label="$2"
  local found
  found="$(find "$MACOS_DIR" -path "$pattern" 2>/dev/null | head -n 1 || true)"
  if [[ -z "$found" ]]; then
    echo "WARN: $label not found in bundle (pattern: $pattern)" >&2
    return 0
  fi
  echo "$label: $found"
  file "$found" | grep -q "$EXPECTED"
}

check_bundle_lib "*/cryptography/hazmat/bindings/_rust.abi3.so" "cryptography"
check_bundle_lib "*/faiss/_swigfaiss*.so" "faiss"
check_bundle_lib "*/torch/lib/libtorch_python.dylib" "torch"

echo "=== Frozen Mac app OK ($EXPECTED) ==="
