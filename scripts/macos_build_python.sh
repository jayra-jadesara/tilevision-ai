#!/usr/bin/env bash
# Run the Mac build Python interpreter (venv with correct CPU architecture).
#
# Requires MACOS_PYTHON_PATH (absolute path to venv python).
#
# Usage:
#   bash scripts/macos_build_python.sh -m pip install ...
#   bash scripts/macos_build_python.sh scripts/download_dinov2_model.py

set -euo pipefail

exec "${MACOS_PYTHON_PATH:?set MACOS_PYTHON_PATH}" "$@"
