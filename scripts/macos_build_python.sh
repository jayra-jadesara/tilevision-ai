#!/usr/bin/env bash
# Run the Mac build Python with the correct CPU architecture.
#
# Requires MACOS_BUILD_ARCH (x64|arm64) and MACOS_PYTHON_PATH (absolute path).
#
# Usage:
#   bash scripts/macos_build_python.sh -m pip install ...
#   bash scripts/macos_build_python.sh scripts/download_dinov2_model.py

set -euo pipefail

PY="${MACOS_PYTHON_PATH:?set MACOS_PYTHON_PATH}"
ARCH="${MACOS_BUILD_ARCH:?set MACOS_BUILD_ARCH}"

if [[ "$ARCH" == "x64" ]]; then
  exec arch -x86_64 "$PY" "$@"
fi

exec "$PY" "$@"
