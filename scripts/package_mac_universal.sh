#!/usr/bin/env bash
# Combine Intel + Apple Silicon Mac DMGs into one zip for all Mac customers.
set -euo pipefail

INTEL_DMG="${1:?Intel dmg path required}"
ARM_DMG="${2:?Apple Silicon dmg path required}"
OUT_ZIP="${3:?Output zip path required}"
INSTALL_TXT="${4:-packaging/MAC_INSTALL.txt}"

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

cp "$INTEL_DMG" "$STAGING/"
cp "$ARM_DMG" "$STAGING/"
cp "$INSTALL_TXT" "$STAGING/READ ME FIRST.txt"

ROOT="$(pwd)"
(
  cd "$STAGING"
  zip -9 -r "$ROOT/$OUT_ZIP" .
)

echo "Created universal Mac package: $OUT_ZIP"
ls -lh "$OUT_ZIP"
