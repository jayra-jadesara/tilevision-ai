"""
Check GPU / CUDA readiness for TileVision AI.

Usage:
    python dev_tools/check_gpu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.gpu_info import detect_gpu_runtime


def main() -> int:
    info = detect_gpu_runtime(preference="auto")

    print("=== TileVision AI — GPU Check ===")
    print(f"PyTorch:     {info.torch_version}")
    print(f"CUDA build:  {info.cuda_version or 'none (CPU wheel)'}")
    print(f"CUDA avail:  {info.is_available}")
    print(f"GPU count:   {info.device_count}")
    if info.device_name:
        print(f"GPU 0:       {info.device_name}")
    if info.vram_gb:
        print(f"VRAM:        {info.vram_gb:.1f} GB")
    print(f"Active:      {info.active_device.upper()}")
    print(f"UI summary:  {info.summary_for_ui()}")

    if not info.using_gpu:
        print()
        print("To enable GPU on Windows:")
        print("  powershell -ExecutionPolicy Bypass -File scripts/install_pytorch_cuda.ps1")
        print("Then restart TileVision AI and run this script again.")
        return 1

    print()
    print("GPU is ready. Indexing and search will use CUDA + mixed precision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
