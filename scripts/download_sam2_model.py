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

Installer bundling (Windows + Mac Apple Silicon, skip Mac Intel by default):

    export TILEVISION_BUNDLE_SAM2=auto
    # then run scripts/build_windows.ps1 or scripts/build_mac.sh
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

# Keep release assets under GitHub's 2 GiB limit (DINOv2 is already ~1.1 GB).
# Prefer safetensors only — skip the duplicate ~149 MB .pt checkpoint.
_MAX_MODEL_BYTES = 220_000_000  # ~210 MB


def bundled_sam2_dir() -> Path:
    return runtime_root() / "model_weights" / _BUNDLED_DIRNAME


def _verify_download(out: Path) -> None:
    config = out / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"config.json missing after download: {out}")
    weights = list(out.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"No .safetensors weight file found in {out}")
    # Drop duplicate native checkpoints if an older download left them behind.
    for leftover in out.glob("*.pt"):
        leftover.unlink(missing_ok=True)
        print(f"Removed duplicate checkpoint: {leftover.name}")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"SAM2 bundle size: {total / (1024**2):.1f} MB ({total} bytes)")
    if total > _MAX_MODEL_BYTES:
        raise RuntimeError(
            f"SAM2 bundle too large for release builds ({total} bytes). "
            "Keep safetensors only and remove .pt / video extras."
        )


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
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "video_preprocessor_config.json",
            "model.safetensors",
            "*.json",
            "*.yaml",
        ],
        ignore_patterns=[
            "*.pt",
            "*.bin",
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
    print("Bundle into installers with: TILEVISION_BUNDLE_SAM2=auto")
    print("Enable Precise Crop in Settings (default ON on this lab branch).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
