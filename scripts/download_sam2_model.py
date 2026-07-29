#!/usr/bin/env python3
"""
Download SAM 2 tiny weights for experimental Precise Crop (lab / optional).

NOT required for production v1.0.12 customer builds.

    python scripts/download_sam2_model.py

Output (default):
    model_weights/sam2.1-hiera-tiny/

Override:

    export TILEVISION_SAM2_MODEL_DIR=/path/to/sam2.1-hiera-tiny
    python scripts/download_sam2_model.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.preprocess.sam2_backend import (  # noqa: E402
    DEFAULT_SAM2_MODEL_ID,
    _BUNDLED_DIRNAME,
)
from src.ai.model_paths import runtime_root  # noqa: E402


def bundled_sam2_dir() -> Path:
    return runtime_root() / "model_weights" / _BUNDLED_DIRNAME


def _verify_download(out: Path) -> None:
    config = out / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"config.json missing after download: {out}")
    weights = list(out.glob("*.safetensors")) + list(out.glob("*.bin"))
    if not weights:
        raise FileNotFoundError(f"No weight files found in {out}")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"SAM2 bundle size: {total / (1024**3):.2f} GB ({total} bytes)")


def main() -> int:
    out = Path(
        os.environ.get("TILEVISION_SAM2_MODEL_DIR", bundled_sam2_dir())
    ).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DEFAULT_SAM2_MODEL_ID} to {out} ...")
    print("(Experimental Precise Crop only — not for default search.)")

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=DEFAULT_SAM2_MODEL_ID,
        local_dir=str(out),
        local_dir_use_symlinks=False,
        ignore_patterns=[
            "*.h5",
            "*.ot",
            "*.msgpack",
            ".git*",
            "*.md",
            "*.jpg",
            "*.png",
        ],
    )
    _verify_download(out)
    print(f"Done. Set TILEVISION_SAM2_MODEL_DIR={out} if needed.")
    print("Enable Precise Crop in Settings (default ON on this lab branch).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
