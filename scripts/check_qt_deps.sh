#!/usr/bin/env bash
# Quick check for common Qt/X11 dependencies on Linux before launching TileVision AI.
#
# Usage:
#   bash scripts/check_qt_deps.sh

set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script is for Linux only."
  exit 0
fi

missing=()

check_pkg() {
  local pkg="$1"
  if command -v dpkg >/dev/null 2>&1; then
    dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
  elif command -v rpm >/dev/null 2>&1; then
    rpm -q "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
  fi
}

for pkg in \
  libxcb-cursor0 \
  libxcb-xinerama0 \
  libxkbcommon-x11-0 \
  libegl1 \
  libglib2.0-0; do
  check_pkg "$pkg"
done

if ((${#missing[@]} == 0)); then
  echo "Common Qt/X11 packages look installed."
  exit 0
fi

echo "Some Qt/X11 packages may be missing on this Linux system:"
printf '  - %s\n' "${missing[@]}"
echo
echo "On Debian/Ubuntu, try:"
echo "  sudo apt update"
echo "  sudo apt install -y libxcb-cursor0 libxcb-xinerama0 libxkbcommon-x11-0 libegl1 libglib2.0-0"
echo
echo "On Fedora/RHEL, package names differ — install the Qt6/XCB runtime libraries for your distro."
