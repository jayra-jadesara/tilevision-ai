#!/usr/bin/env bash
# Install Mac build dependencies into an isolated venv with correct CPU architecture.
#
# CI:
#   Intel  → macos-15-intel runner (native x86_64 — no Rosetta)
#   arm64  → macos-latest runner (native Apple Silicon)
#
# Usage:
#   PYTHON_SETUP_PATH=/path/to/python source scripts/install_mac_deps.sh x64
#   PYTHON_SETUP_PATH=/path/to/python source scripts/install_mac_deps.sh arm64
#
# Sets: MACOS_BUILD_ARCH, MACOS_PYTHON_PATH

set -euo pipefail

MACOS_BUILD_ARCH="${1:?Usage: install_mac_deps.sh x64|arm64}"
PY="${PYTHON_SETUP_PATH:?set PYTHON_SETUP_PATH to setup-python output}"

if [[ "$MACOS_BUILD_ARCH" == "x64" ]]; then
  EXPECT_ARCH="x86_64"
  VENV="$(pwd)/.venv-macos-x64"
elif [[ "$MACOS_BUILD_ARCH" == "arm64" ]]; then
  EXPECT_ARCH="arm64"
  VENV="$(pwd)/.venv-macos-arm64"
else
  echo "ERROR: MACOS_BUILD_ARCH must be x64 or arm64" >&2
  exit 1
fi

echo "=== Installing Mac ${MACOS_BUILD_ARCH} deps (expect ${EXPECT_ARCH}) ==="
file "$PY"

rm -rf "$VENV"
"$PY" -m venv "$VENV"
MACOS_PYTHON_PATH="$VENV/bin/python"

"$MACOS_PYTHON_PATH" -c "
import platform
machine = platform.machine()
print('python machine:', machine)
expected = '${EXPECT_ARCH}'
if expected == 'x86_64':
    assert machine in ('x86_64', 'AMD64'), machine
else:
    assert machine == 'arm64', machine
"

HOST_ARCH="$(uname -m)"
if [[ "$MACOS_BUILD_ARCH" == "x64" && "$HOST_ARCH" == "arm64" ]]; then
  echo "ERROR: Intel (x64) Mac builds require a native x86_64 runner (macos-15-intel in CI)." >&2
  echo "Rosetta cross-builds compile native extensions as arm64 and will fail verification." >&2
  exit 1
fi

"$MACOS_PYTHON_PATH" -m pip install --upgrade pip
"$MACOS_PYTHON_PATH" -m pip install --no-cache-dir "numpy>=1.24.0,<2.0.0"
# Never compile Rust/C extensions in CI — use published wheels only.
"$MACOS_PYTHON_PATH" -m pip install --no-cache-dir \
  --only-binary cryptography,faiss-cpu,torch,torchvision,tokenizers,safetensors,opencv-python-headless \
  -r requirements.txt pyinstaller

verify_native_arch() {
  local expected="$1"
  local rust_so
  rust_so="$("$MACOS_PYTHON_PATH" -c "
import pathlib
import cryptography
print(next(pathlib.Path(cryptography.__file__).parent.rglob('_rust.abi3.so')))
")"
  echo "cryptography: $rust_so"
  file "$rust_so"
  if ! file "$rust_so" | grep -qE "${expected}|universal"; then
    echo "ERROR: cryptography is wrong CPU arch (need ${expected})" >&2
    exit 1
  fi
}

verify_native_arch "$EXPECT_ARCH"

export MACOS_BUILD_ARCH MACOS_PYTHON_PATH
echo "MACOS_BUILD_ARCH=$MACOS_BUILD_ARCH"
echo "MACOS_PYTHON_PATH=$MACOS_PYTHON_PATH"
