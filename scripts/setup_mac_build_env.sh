#!/usr/bin/env bash
# Create an isolated macOS build venv for Intel (x64) or Apple Silicon (arm64).
#
# Usage (CI and local):
#   source scripts/setup_mac_build_env.sh x64   # Intel Mac customers
#   source scripts/setup_mac_build_env.sh arm64 # M1/M2/M3/M4 Mac customers
#
# Requires: python-path from actions/setup-python OR python3 on PATH (local).
# Sets: MACOS_PYTHON, MACOS_VENV, MACOS_ARCH, MACOS_VERIFY_ARCH

set -euo pipefail

MACOS_ARCH="${1:?Usage: setup_mac_build_env.sh x64|arm64}"
case "$MACOS_ARCH" in
  x64)
    MACOS_VERIFY_ARCH="x86_64"
    ;;
  arm64)
    MACOS_VERIFY_ARCH="arm64"
    ;;
  *)
    echo "ERROR: MACOS_ARCH must be x64 or arm64 (got: $MACOS_ARCH)" >&2
    exit 1
    ;;
esac

if [[ -n "${PYTHON_SETUP_PATH:-}" ]]; then
  PY="$PYTHON_SETUP_PATH"
elif [[ -n "${pythonLocation:-}" ]]; then
  PY="${pythonLocation}/bin/python3"
else
  PY="$(command -v python3)"
fi

MACOS_VENV=".venv-macos-${MACOS_ARCH}"
echo "setup_mac_build_env: arch=$MACOS_ARCH python=$PY venv=$MACOS_VENV"
file "$PY"

if [[ "$MACOS_ARCH" == "x64" ]]; then
  arch -x86_64 "$PY" -m venv "$MACOS_VENV"
  MACOS_PYTHON="arch -x86_64 $MACOS_VENV/bin/python"
else
  "$PY" -m venv "$MACOS_VENV"
  MACOS_PYTHON="$MACOS_VENV/bin/python"
fi

export MACOS_PYTHON MACOS_VENV MACOS_ARCH MACOS_VERIFY_ARCH
