#!/usr/bin/env bash
# Install CUDA-enabled PyTorch on Linux (NVIDIA GPU required).
#
# Usage:
#   bash scripts/install_pytorch_cuda.sh
#
# Requires: Python 3.12+, pip, NVIDIA driver, and an NVIDIA GPU.

set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No NVIDIA GPU detected (nvidia-smi not found)."
  echo "TileVision AI will use CPU inference on this machine."
  exit 0
fi

echo "Uninstalling CPU-only PyTorch wheels (if present)..."
python3 -m pip uninstall -y torch torchvision torchaudio || true

echo "Installing CUDA PyTorch (cu124 index)..."
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

python3 - <<'PY'
import torch

if torch.cuda.is_available():
    print(f"CUDA PyTorch active on {torch.cuda.get_device_name(0)}.")
else:
    print("CUDA PyTorch installed but GPU is still inactive.")
    print("Install the latest NVIDIA driver, then restart TileVision AI.")
    raise SystemExit(1)
PY
