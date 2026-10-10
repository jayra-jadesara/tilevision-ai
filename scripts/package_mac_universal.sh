#!/usr/bin/env bash
# Zip BOTH arch-specific DMGs for vendors who want one download.
# This is NOT a Universal2 .app — customers still pick Intel vs Apple Silicon.
set -euo pipefail

INTEL_DMG="${1:?Intel DMG path}"
ARM_DMG="${2:?Apple Silicon DMG path}"
OUT_ZIP="${3:?output zip path}"

mkdir -p "$(dirname "$OUT_ZIP")"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/tv-both.XXXXXX")"
cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

cp "$INTEL_DMG" "$STAGE/"
cp "$ARM_DMG" "$STAGE/"
cp packaging/MAC_INSTALL.txt "$STAGE/READ ME FIRST.txt"
cat >"$STAGE/NOT_UNIVERSAL2.txt" <<'EOF'
These are TWO separate builds (Intel + Apple Silicon).

TileVision AI does not ship a Universal2 .app because Mac Intel requires
torch 2.2.2 wheels while Apple Silicon uses newer torch — those native
libraries cannot be combined with lipo.

Open READ ME FIRST.txt and pick the DMG that matches your Mac.
EOF

rm -f "$OUT_ZIP"
(
  cd "$STAGE"
  zip -9 -r "$OUT_ZIP" .
)
echo "Wrote $OUT_ZIP"
ls -lh "$OUT_ZIP"
