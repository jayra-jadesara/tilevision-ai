#!/usr/bin/env bash
# Create a professional macOS DMG for TileVision AI.
#
# Layout:
#   TileVision AI.app
#   Applications → /Applications  (symlink)
# Volume name: TileVision AI
#
# Usage:
#   bash scripts/create_mac_dmg.sh "dist/TileVision AI.app" dist/TileVision-AI-Intel.dmg
#
# Optional 3rd arg overrides volume name (default: "TileVision AI").
set -euo pipefail

APP_PATH="${1:?app bundle path required}"
OUTPUT_DMG="${2:?output .dmg path required}"
VOLNAME="${3:-TileVision AI}"

if [[ ! -d "$APP_PATH" ]]; then
  echo "ERROR: app bundle not found: $APP_PATH" >&2
  exit 1
fi

if ! command -v hdiutil >/dev/null 2>&1; then
  echo "ERROR: hdiutil not available (macOS only)" >&2
  exit 1
fi

echo "=== Create professional macOS DMG ==="
echo "app:    $APP_PATH"
echo "output: $OUTPUT_DMG"
echo "volume: $VOLNAME"
df -h .

APP_MB="$(du -sm "$APP_PATH" | awk '{print $1}')"
echo "App bundle size: ${APP_MB} MB"

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/tilevision-dmg.XXXXXX")"
cleanup() {
  rm -rf "$STAGE"
  if [[ -n "${ATTACHED_DEV:-}" ]]; then
    hdiutil detach "$ATTACHED_DEV" -force >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

APP_NAME="$(basename "$APP_PATH")"
ditto "$APP_PATH" "$STAGE/$APP_NAME"
ln -s /Applications "$STAGE/Applications"
# Customer-facing install note (unsigned Gatekeeper steps).
if [[ -f packaging/MAC_INSTALL.txt ]]; then
  cp packaging/MAC_INSTALL.txt "$STAGE/READ ME FIRST.txt"
fi

rm -f "$OUTPUT_DMG" "${OUTPUT_DMG%.dmg}.temp.dmg"

STAGE_MB="$(du -sm "$STAGE" | awk '{print $1}')"
# HFS+ needs substantial slack beyond `du` for large bundles with many small files.
# Previous APP_MB+280 overflowed at ~3 GiB apps ("No space left on device" on the volume).
SIZE_MB=$(( STAGE_MB + STAGE_MB / 4 + 768 ))
if (( SIZE_MB < STAGE_MB + 1024 )); then
  SIZE_MB=$((STAGE_MB + 1024))
fi
TMP_DMG="${OUTPUT_DMG%.dmg}.temp.dmg"

echo "Staged size: ${STAGE_MB} MB → RW image ${SIZE_MB} MB"

# Prefer auto-sized srcfolder image (correct HFS+ sizing). Fall back to explicit -size.
set +e
hdiutil create -volname "$VOLNAME" -srcfolder "$STAGE" -ov -format UDRW "$TMP_DMG"
CREATE_RC=$?
set -e
if [[ $CREATE_RC -ne 0 || ! -f "$TMP_DMG" ]]; then
  echo "WARN: hdiutil -srcfolder failed (rc=$CREATE_RC); retrying with explicit size ${SIZE_MB}m"
  rm -f "$TMP_DMG"
  hdiutil create -size "${SIZE_MB}m" -fs HFS+ -volname "$VOLNAME" -ov "$TMP_DMG"
  echo "Attaching sized image for copy..."
  ATTACH_OUT="$(hdiutil attach -readwrite -noverify -noautoopen "$TMP_DMG")"
  echo "$ATTACH_OUT"
  ATTACHED_DEV="$(echo "$ATTACH_OUT" | awk '/\/dev\// {print $1; exit}')"
  MOUNT_DIR="$(echo "$ATTACH_OUT" | awk -F'\t' '/\/Volumes\// {print $NF; exit}')"
  if [[ -z "$MOUNT_DIR" ]]; then
    MOUNT_DIR="$(echo "$ATTACH_OUT" | grep -o '/Volumes/[^ ]*' | tail -1 || true)"
  fi
  if [[ -z "$ATTACHED_DEV" || -z "$MOUNT_DIR" || ! -d "$MOUNT_DIR" ]]; then
    echo "ERROR: failed to resolve DMG mount (dev=$ATTACHED_DEV mount=$MOUNT_DIR)" >&2
    exit 1
  fi
  find "$MOUNT_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  ditto "$STAGE" "$MOUNT_DIR"
  sync
  hdiutil detach "$ATTACHED_DEV"
  ATTACHED_DEV=""
fi

echo "Attaching for Finder layout..."
ATTACH_OUT="$(hdiutil attach -readwrite -noverify -noautoopen "$TMP_DMG")"
echo "$ATTACH_OUT"
ATTACHED_DEV="$(echo "$ATTACH_OUT" | awk '/\/dev\// {print $1; exit}')"
MOUNT_DIR="$(echo "$ATTACH_OUT" | awk -F'\t' '/\/Volumes\// {print $NF; exit}')"
if [[ -z "$MOUNT_DIR" ]]; then
  MOUNT_DIR="$(echo "$ATTACH_OUT" | grep -o '/Volumes/[^ ]*' | tail -1 || true)"
fi
if [[ -z "$ATTACHED_DEV" || -z "$MOUNT_DIR" || ! -d "$MOUNT_DIR" ]]; then
  echo "ERROR: failed to resolve DMG mount for layout (dev=$ATTACHED_DEV mount=$MOUNT_DIR)" >&2
  exit 1
fi

# Best-effort Finder window layout (ignored if AppleScript unavailable).
if command -v osascript >/dev/null 2>&1; then
  osascript <<EOF || true
tell application "Finder"
  tell disk "$VOLNAME"
    open
    set current view of container window to icon view
    set toolbar visible of container window to false
    set statusbar visible of container window to false
    set the bounds of container window to {120, 120, 780, 480}
    set viewOptions to the icon view options of container window
    set arrangement of viewOptions to not arranged
    set icon size of viewOptions to 96
    set position of item "$APP_NAME" of container window to {160, 180}
    set position of item "Applications" of container window to {480, 180}
    try
      set position of item "READ ME FIRST.txt" of container window to {320, 340}
    end try
    update without registering applications
    delay 1
    close
  end tell
end tell
EOF
fi

sync
hdiutil detach "$ATTACHED_DEV"
ATTACHED_DEV=""

echo "Compressing UDZO..."
hdiutil convert "$TMP_DMG" -format UDZO -imagekey zlib-level=9 -o "$OUTPUT_DMG"
rm -f "$TMP_DMG"

echo "DMG OK: $OUTPUT_DMG"
ls -lh "$OUTPUT_DMG"
file "$OUTPUT_DMG"
