#!/usr/bin/env python3
"""
Download SAM 2.1 tiny ONNX weights for Mac Intel + Windows Precise Crop.

Works with production torch (including Mac Intel 2.2.x) via onnxruntime.

    python scripts/download_sam2_onnx_model.py

Output:
    model_weights/sam2.1-hiera-tiny-onnx/
      sam2.1_hiera_tiny.encoder.onnx
      sam2.1_hiera_tiny.decoder.onnx

Installer bundling:

    export TILEVISION_BUNDLE_SAM2=auto   # includes Mac Intel ONNX
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.preprocess.sam2_onnx_backend import (  # noqa: E402
    BUNDLED_ONNX_DIRNAME,
    DEFAULT_ONNX_REPO,
    DEFAULT_ONNX_ZIP,
)
from src.ai.model_paths import runtime_root  # noqa: E402

_MAX_BYTES = 180_000_000  # ~170 MB (encoder+decoder ~126 MB)


def bundled_onnx_dir() -> Path:
    return runtime_root() / "model_weights" / BUNDLED_ONNX_DIRNAME


def _verify(out: Path) -> None:
    enc = next(out.glob("*.encoder.onnx"), None)
    dec = next(out.glob("*.decoder.onnx"), None)
    if enc is None or dec is None:
        raise FileNotFoundError(f"encoder/decoder ONNX missing in {out}")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"ONNX SAM2 bundle size: {total / (1024**2):.1f} MB ({total} bytes)")
    if total > _MAX_BYTES:
        raise RuntimeError(f"ONNX bundle too large: {total} bytes")


def main() -> int:
    out = Path(
        os.environ.get("TILEVISION_SAM2_ONNX_DIR", bundled_onnx_dir())
    ).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DEFAULT_ONNX_REPO}/{DEFAULT_ONNX_ZIP} ...")
    print(f"Extracting to {out}")

    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(repo_id=DEFAULT_ONNX_REPO, filename=DEFAULT_ONNX_ZIP)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)

    _verify(out)
    print(f"Done. Set TILEVISION_SAM2_ONNX_DIR={out} if needed.")
    print("Mac Intel + Windows Precise Crop will use ONNX SAM2 when enabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
