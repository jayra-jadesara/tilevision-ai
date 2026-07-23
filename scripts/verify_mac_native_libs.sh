#!/usr/bin/env bash
# Verify Python and critical native extensions match the target Mac CPU architecture.

set -euo pipefail

EXPECTED="${1:?expected arch: x86_64 or arm64}"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

echo "=== Verifying Mac build environment ($EXPECTED) ==="

bash scripts/macos_build_python.sh -c "
import pathlib
import platform

machine = platform.machine()
print('python machine:', machine)
expected = '${EXPECTED}'
if expected == 'x86_64':
    assert machine in ('x86_64', 'AMD64'), machine
else:
    assert machine == 'arm64', machine

import cryptography
import faiss
import torch
import PySide6.QtCore  # noqa: F401
import numpy as np

print('numpy:', np.__version__)
print('imports ok: cryptography, faiss, torch, PySide6')
print('torch:', torch.__version__)

rust = next(pathlib.Path(cryptography.__file__).parent.rglob('_rust.abi3.so'))
print('cryptography_rust:', rust)
"

RUST_SO="$(bash scripts/macos_build_python.sh -c "
import pathlib, cryptography
print(next(pathlib.Path(cryptography.__file__).parent.rglob('_rust.abi3.so')))
")"
file "$RUST_SO"
file "$RUST_SO" | grep -qE "${EXPECTED}|universal"

echo "=== Mac build environment OK ($EXPECTED) ==="
