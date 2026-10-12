#!/usr/bin/env python3
"""
Download CLIP ViT-B/32 weights for TileVision AI description search.

Separate from DINOv2 — keep this as its own asset so the main installer
stays under GitHub's 2 GiB per-file limit. Customers who never use text
search can skip this entirely (lazy first-use download also supported).

    python scripts/download_clip_model.py

Output (default):
    model_weights/clip-vit-base-patch32/

Override:

    export TILEVISION_CLIP_MODEL_DIR=/path/to/clip-vit-base-patch32
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.clip_model_paths import (  # noqa: E402
    DEFAULT_CLIP_MODEL_ID,
    bundled_clip_model_dir,
)

# CLIP ViT-B/32 is ~350–600 MB — far under the 2 GiB GitHub asset cap when
# shipped as a *separate* download (not merged into the DINOv2 installer).
_MAX_MODEL_BYTES = 800_000_000  # ~760 MB hard ceiling
_MAX_ATTEMPTS = int(os.environ.get("TILEVISION_HF_DOWNLOAD_ATTEMPTS", "6"))


def _verify_download(out: Path) -> None:
    config = out / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"config.json missing after download: {out}")
    weights = list(out.glob("*.safetensors")) + list(out.glob("*.bin"))
    if not weights:
        raise FileNotFoundError(f"No weight file found in {out}")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"CLIP bundle size: {total / (1024**2):.1f} MB ({total} bytes)")
    if total > _MAX_MODEL_BYTES:
        raise RuntimeError(
            f"CLIP bundle too large ({total} bytes > {_MAX_MODEL_BYTES}). "
            "Tighten allow_patterns or use a smaller checkpoint."
        )


def main() -> int:
    out = Path(
        os.environ.get("TILEVISION_CLIP_MODEL_DIR", bundled_clip_model_dir())
    ).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    if (out / "config.json").is_file() and (
        list(out.glob("*.safetensors")) or list(out.glob("*.bin"))
    ):
        print(f"CLIP weights already present at {out} — skipping download.")
        _verify_download(out)
        return 0

    print(f"Downloading {DEFAULT_CLIP_MODEL_ID} to {out} ...")
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    last_exc: BaseException | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            snapshot_download(
                repo_id=DEFAULT_CLIP_MODEL_ID,
                local_dir=str(out),
                local_dir_use_symlinks=False,
                allow_patterns=[
                    "config.json",
                    "preprocessor_config.json",
                    "tokenizer*",
                    "vocab.json",
                    "merges.txt",
                    "special_tokens_map.json",
                    "model.safetensors",
                    "pytorch_model.bin",
                    "*.json",
                ],
                # Prefer safetensors when present; otherwise pytorch_model.bin
                # (openai/clip-vit-base-patch32 ships .bin only on the Hub).
                ignore_patterns=["*.h5", "*.msgpack", "flax_model*", "tf_model*", ".git*", "*.md"],
            )
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            retryable = (
                isinstance(exc, HfHubHTTPError)
                or "429" in str(exc)
                or "Too Many Requests" in str(exc)
            )
            if attempt >= _MAX_ATTEMPTS or not retryable:
                raise
            delay = min(120, 8 * (2 ** (attempt - 1)))
            print(
                f"Hub download attempt {attempt}/{_MAX_ATTEMPTS} failed ({exc}). "
                f"Retrying in {delay}s..."
            )
            time.sleep(delay)
    if last_exc is not None:
        raise last_exc

    # snapshot_download sometimes skips large weight blobs when mixing
    # allow_patterns; pull the weight file explicitly if still missing.
    weights = list(out.glob("*.safetensors")) + list(out.glob("*.bin"))
    if not weights:
        from huggingface_hub import hf_hub_download

        for name in ("model.safetensors", "pytorch_model.bin"):
            try:
                hf_hub_download(
                    repo_id=DEFAULT_CLIP_MODEL_ID,
                    filename=name,
                    local_dir=str(out),
                )
                break
            except Exception as exc:
                print(f"Weight candidate {name} unavailable: {exc}")

    _verify_download(out)
    print()
    print("Done.")
    print(f"  Weights saved to: {out}")
    print("  Ship separately from the DINOv2 installer (lazy first-use OK).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
