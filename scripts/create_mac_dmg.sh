#!/usr/bin/env bash
# Create a compressed macOS DMG from a .app bundle (CI + local builds).
#
# Usage:
#   bash scripts/create_mac_dmg.sh dist/TileVisionAI.app dist/out.dmg "TileVision AI (Intel)"
set -euo pipefail

APP_PATH="${1:?app bundle path required}"
OUTPUT_DMG="${2:?output .dmg path required}"
VOLNAME="${3:?volume name required}"

if [[ ! -d "$APP_PATH" ]]; then
  echo "ERROR: app bundle not found: $APP_PATH" >&2
  exit 1
fi

echo "=== Create macOS DMG ==="
echo "app:    $APP_PATH"
echo "output: $OUTPUT_DMG"
echo "volume: $VOLNAME"
df -h .

APP_MB="$(du -sm "$APP_PATH" | awk '{print $1}')"
echo "App bundle size: ${APP_MB} MB"

rm -f "$OUTPUT_DMG" "${OUTPUT_DMG%.dmg}.temp.dmg"

create_dmg_direct() {
  hdiutil create -volname "$VOLNAME" -srcfolder "$APP_PATH" -ov -format UDZO "$OUTPUT_DMG"
}

create_dmg_staged() {
  local size_mb tmp_dmg dev mount_dir app_name
  size_mb=$((APP_MB + 256))
  tmp_dmg="${OUTPUT_DMG%.dmg}.temp.dmg"
  app_name="$(basename "$APP_PATH")"

  echo "Staged DMG create (${size_mb} MB image)..."
  hdiutil create -size "${size_mb}m" -volname "$VOLNAME" -fs HFS+ -format UDRW -ov "$tmp_dmg"

  dev="$(hdiutil attach -readwrite -noverify "$tmp_dmg" | awk '/Apple_HFS/ {print $1; exit}')"
  if [[ -z "$dev" ]]; then
    echo "ERROR: failed to attach temporary DMG" >&2
    exit 1
  fi

  mount_dir="$(hdiutil info | awk -v dev="$dev" '$0 ~ dev {print $3; exit}')"
  if [[ -z "$mount_dir" || ! -d "$mount_dir" ]]; then
    echo "ERROR: could not resolve DMG mount path" >&2
    hdiutil detach "$dev" || true
    exit 1
  fi

  ditto "$APP_PATH" "$mount_dir/$app_name"
  hdiutil detach "$dev"
  hdiutil convert "$tmp_dmg" -format UDZO -imagekey zlib-level=9 -o "$OUTPUT_DMG"
  rm -f "$tmp_dmg"
}

for attempt in 1 2 3; do
  echo "DMG attempt $attempt/3 (direct hdiutil)..."
  if create_dmg_direct; then
    echo "DMG OK (direct): $OUTPUT_DMG"
    ls -lh "$OUTPUT_DMG"
    exit 0
  fi
  echo "Direct create failed — waiting before retry..."
  rm -f "$OUTPUT_DMG"
  sleep "$attempt"
done

create_dmg_staged
echo "DMG OK (staged): $OUTPUT_DMG"
ls -lh "$OUTPUT_DMG"
