#!/usr/bin/env bash
# Free disk space on GitHub-hosted macOS runners before large PyInstaller + DMG builds.
set -euo pipefail

echo "Disk before cleanup:"
df -h .

# Safe removals commonly used on GHA macOS images (not needed for the build itself).
sudo rm -rf \
  /usr/share/dotnet \
  /usr/local/lib/android \
  /opt/hostedtoolcache/CodeQL \
  /Users/runner/Library/Caches/Homebrew \
  /Users/runner/.nuget \
  /Users/runner/hostedtoolcache/node \
  /Users/runner/hostedtoolcache/go \
  || true

# Local build leftovers (idempotent).
rm -rf build dist/TileVisionAI 2>/dev/null || true

echo "Disk after cleanup:"
df -h .
