#!/usr/bin/env bash
# Deprecated — use scripts/install_mac_deps.sh instead.
#
# Kept for backward compatibility. Sources install_mac_deps.sh and exports
# legacy variable names used by older scripts.

set -euo pipefail

MACOS_ARCH="${1:?Usage: setup_mac_build_env.sh x64|arm64}"
# shellcheck disable=SC1091
source "$(dirname "$0")/install_mac_deps.sh" "$MACOS_ARCH"

MACOS_VENV=".venv-macos-${MACOS_ARCH}"
MACOS_VERIFY_ARCH="$([[ "$MACOS_ARCH" == x64 ]] && echo x86_64 || echo arm64)"
export MACOS_VENV MACOS_VERIFY_ARCH
