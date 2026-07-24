#!/usr/bin/env python3
"""
Download DINOv2-large weights for offline TileVision AI installs.

Run once on a build machine with internet access:

    python scripts/download_dinov2_model.py

Output (default):
    model_weights/dinov2-large/

Override destination:

    set TILEVISION_MODEL_DIR=C:\\path\\to\\dinov2-large   # Windows
    export TILEVISION_MODEL_DIR=/path/to/dinov2-large      # macOS / Linux
    python scripts/download_dinov2_model.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.model_paths import DEFAULT_MODEL_ID, bundled_model_dir  # noqa: E402


def _verify_download(out: Path) -> None:
    config = out / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"config.json missing after download: {out}")
    # At least one weight file (safetensors or bin)
    weights = list(out.glob("*.safetensors")) + list(out.glob("pytorch_model*.bin"))
    if not weights:
        raise FileNotFoundError(f"No model weight files found in {out}")


def main() -> int:
    out = Path(os.environ.get("TILEVISION_MODEL_DIR", bundled_model_dir())).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DEFAULT_MODEL_ID} to {out} ...")
    print("(This is ~1 GB — may take several minutes.)")

    # huggingface_hub only — no torch/transformers import required (works on Mac Intel x64).
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=DEFAULT_MODEL_ID,
        local_dir=str(out),
        local_dir_use_symlinks=False,
    )
    _verify_download(out)

    print()
    print("Done.")
    print(f"  Weights saved to: {out}")
    print()
    print("For offline customer builds, bundle this folder with PyInstaller")
    print("(see packaging/README.md) or set TILEVISION_MODEL_DIR to this path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
