#!/usr/bin/env bash
# Install Mac build dependencies with correct CPU architecture.
#
# Intel (x64): uses setup-python x64 directly under Rosetta — NO venv.
#   (venv on Apple Silicon runners pulls arm64 wheels into x86_64 builds)
#
# Apple Silicon (arm64): uses setup-python arm64 with isolated venv.
#
# Usage:
#   PYTHON_SETUP_PATH=/path/to/python source scripts/install_mac_deps.sh x64
#   PYTHON_SETUP_PATH=/path/to/python source scripts/install_mac_deps.sh arm64
#
# Sets: MACOS_BUILD_ARCH, MACOS_PYTHON_PATH

set -euo pipefail

MACOS_BUILD_ARCH="${1:?Usage: install_mac_deps.sh x64|arm64}"
PY="${PYTHON_SETUP_PATH:?set PYTHON_SETUP_PATH to setup-python output}"

run_py() {
  if [[ "$MACOS_BUILD_ARCH" == "x64" ]]; then
    arch -x86_64 "$PY" "$@"
  else
    "$PY" "$@"
  fi
}

verify_rust_arch() {
  local expected="$1"
  local rust_so
  rust_so="$(run_py -c "
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

if [[ "$MACOS_BUILD_ARCH" == "x64" ]]; then
  echo "=== Installing Mac Intel (x86_64) deps — direct x64 Python, no venv ==="
  file "$PY"
  run_py -c "import platform; assert platform.machine() in ('x86_64','AMD64'), platform.machine()"

  MACOS_PYTHON_PATH="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"

  run_py -m pip install --upgrade pip
  run_py -m pip install --no-cache-dir "numpy>=1.24.0,<2.0.0"
  run_py -m pip install --no-cache-dir -r requirements.txt pyinstaller

  for pkg in cryptography faiss-cpu; do
    run_py -m pip install --no-cache-dir --force-reinstall --no-deps "$pkg"
  done

  verify_rust_arch "x86_64"

elif [[ "$MACOS_BUILD_ARCH" == "arm64" ]]; then
  echo "=== Installing Mac Apple Silicon (arm64) deps ==="
  file "$PY"
  run_py -c "import platform; assert platform.machine() == 'arm64', platform.machine()"

  VENV="$(pwd)/.venv-macos-arm64"
  "$PY" -m venv "$VENV"
  MACOS_PYTHON_PATH="$VENV/bin/python"

  "$MACOS_PYTHON_PATH" -m pip install --upgrade pip
  "$MACOS_PYTHON_PATH" -m pip install --no-cache-dir "numpy>=1.24.0,<2.0.0"
  "$MACOS_PYTHON_PATH" -m pip install --no-cache-dir -r requirements.txt pyinstaller

  verify_rust_arch "arm64"

else
  echo "ERROR: MACOS_BUILD_ARCH must be x64 or arm64" >&2
  exit 1
fi

export MACOS_BUILD_ARCH MACOS_PYTHON_PATH
echo "MACOS_BUILD_ARCH=$MACOS_BUILD_ARCH"
echo "MACOS_PYTHON_PATH=$MACOS_PYTHON_PATH"
