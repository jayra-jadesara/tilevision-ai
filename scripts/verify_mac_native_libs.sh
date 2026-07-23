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

echo "=== Verifying Mac build environment ($EXPECTED) ==="

$MACOS_PYTHON -c "import platform; print('python machine:', platform.machine())"
if [[ "$EXPECTED" == "x86_64" ]]; then
  $MACOS_PYTHON -c "import platform; assert platform.machine() in ('x86_64', 'AMD64'), platform.machine()"
else
  $MACOS_PYTHON -c "import platform; assert platform.machine() == 'arm64', platform.machine()"
fi

# Import smoke test — catches wrong-arch wheels before PyInstaller spends 10+ minutes.
$MACOS_PYTHON -c "
import cryptography
import faiss
import torch
import PySide6.QtCore
print('imports ok: cryptography, faiss, torch, PySide6')
print('torch:', torch.__version__, 'mps:', getattr(torch.backends.mps, 'is_available', lambda: False)())
"

check_lib() {
  local pattern="$1"
  local label="$2"
  local found
  found="$(find "$MACOS_VENV" -path "$pattern" 2>/dev/null | head -n 1 || true)"
  if [[ -z "$found" ]]; then
    echo "WARN: $label not found (pattern: $pattern)" >&2
    return 0
  fi
  echo "$label: $found"
  file "$found"
  file "$found" | grep -q "$EXPECTED"
}

check_lib "*/cryptography/hazmat/bindings/_rust.abi3.so" "cryptography"
check_lib "*/faiss/_swigfaiss*.so" "faiss"
check_lib "*/torch/lib/libtorch_python.dylib" "torch"
check_lib "*/PySide6/QtCore.abi3.so" "PySide6"

echo "=== All native libs match $EXPECTED ==="
