"""
CLIP model path resolution for text / description search.

Separate from DINOv2 so the visual-search installer stays under GitHub's
2 GiB asset limit. CLIP is downloaded on first use of text search (or via
``scripts/download_clip_model.py`` for offline vendor builds).
"""

from __future__ import annotations

import os
from pathlib import Path

from src.ai.model_paths import runtime_root

# OpenAI CLIP ViT-B/32 — ~350–600 MB safetensors; HF model card license: MIT.
# Suitable size for a *separate* downloadable asset (not bundled with DINOv2).
DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
_BUNDLED_DIRNAME = "clip-vit-base-patch32"
CLIP_EMBEDDING_DIM = 512


def bundled_clip_model_dir() -> Path:
    return runtime_root() / "model_weights" / _BUNDLED_DIRNAME


def _is_valid_local_model(path: Path) -> bool:
    if not path.is_dir():
        return False
    config = path / "config.json"
    weights = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    return config.is_file() and bool(weights)


def resolve_clip_model_source() -> tuple[str, bool]:
    """
    Return (model_source, local_files_only).

    Prefers ``TILEVISION_CLIP_MODEL_DIR``, then bundled weights, else Hub id
    (unless ``TILEVISION_OFFLINE_MODEL=1``).
    """
    env_dir = os.environ.get("TILEVISION_CLIP_MODEL_DIR", "").strip()
    if env_dir:
        local = Path(env_dir).expanduser()
        if _is_valid_local_model(local):
            return str(local), True
        raise FileNotFoundError(
            f"TILEVISION_CLIP_MODEL_DIR is set but no CLIP weights were found at: {local}"
        )

    bundled = bundled_clip_model_dir()
    if _is_valid_local_model(bundled):
        return str(bundled), True

    offline = os.environ.get("TILEVISION_OFFLINE_MODEL", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if offline:
        raise FileNotFoundError(
            "TILEVISION_OFFLINE_MODEL is set but no CLIP weights were found. "
            f"Run: python scripts/download_clip_model.py "
            f"(expected at {bundled})"
        )

    return DEFAULT_CLIP_MODEL_ID, False


def clip_model_available_locally() -> bool:
    try:
        _source, local_only = resolve_clip_model_source()
        return local_only
    except FileNotFoundError:
        return False
