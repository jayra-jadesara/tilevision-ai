#!/usr/bin/env bash
# Verify Python and critical native extensions match the target Mac CPU architecture.
#
# Usage:
#   MACOS_PYTHON='arch -x86_64 .venv-macos-x64/bin/python' \
#   MACOS_VENV='.venv-macos-x64' \
#   scripts/verify_mac_native_libs.sh x86_64

set -euo pipefail

EXPECTED="${1:?expected arch: x86_64 or arm64}"
MACOS_PYTHON="${MACOS_PYTHON:?set MACOS_PYTHON}"
MACOS_VENV="${MACOS_VENV:?set MACOS_VENV}"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

echo "=== Verifying Mac build environment ($EXPECTED) ==="
echo "venv: $MACOS_VENV"

$MACOS_PYTHON -c "
import platform
import sys

machine = platform.machine()
print('python machine:', machine)
expected = '${EXPECTED}'
if expected == 'x86_64':
    assert machine in ('x86_64', 'AMD64'), machine
else:
    assert machine == 'arm64', machine

# If these imports work, native wheels match this Python architecture.
import cryptography
import faiss
import torch
import PySide6.QtCore  # noqa: F401

print('imports ok: cryptography, faiss, torch, PySide6')
print('torch:', torch.__version__)
if expected == 'arm64':
    print('mps available:', torch.backends.mps.is_available())
"

# Optional: spot-check cryptography .so (most common cross-arch mistake).
RUST_SO="$(find "$MACOS_VENV" -path '*/cryptography/hazmat/bindings/_rust.abi3.so' 2>/dev/null | head -n 1 || true)"
if [[ -n "$RUST_SO" ]]; then
  echo "cryptography: $RUST_SO"
  file "$RUST_SO"
  file "$RUST_SO" | grep -qE "${EXPECTED}|universal"
fi

echo "=== Mac build environment OK ($EXPECTED) ==="
