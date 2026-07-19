"""
DINOv2 model path resolution for TileVision AI.

Supports:
  - Bundled weights in ``model_weights/dinov2-large/`` (PyInstaller / offline installs)
  - ``TILEVISION_MODEL_DIR`` environment variable
  - Hugging Face hub download (development / first run with internet)
  - ``TILEVISION_OFFLINE_MODEL=1`` to require a local copy (no download)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_MODEL_ID = "facebook/dinov2-large"
_BUNDLED_DIRNAME = "dinov2-large"


def runtime_root() -> Path:
    """Project root in dev; PyInstaller extract dir when frozen."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """Repository root (dev only)."""
    return Path(__file__).resolve().parents[2]


def bundled_model_dir() -> Path:
    return runtime_root() / "model_weights" / _BUNDLED_DIRNAME


def _is_valid_local_model(path: Path) -> bool:
    if not path.is_dir():
        return False
    config = path / "config.json"
    weights = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    return config.is_file() and bool(weights)


def resolve_dinov2_model_source() -> tuple[str, bool]:
    """
    Return (model_source, local_files_only).

    ``model_source`` is a Hugging Face model id or a local directory path.
    """
    env_dir = os.environ.get("TILEVISION_MODEL_DIR", "").strip()
    if env_dir:
        local = Path(env_dir).expanduser()
        if _is_valid_local_model(local):
            return str(local), True
        raise FileNotFoundError(
            f"TILEVISION_MODEL_DIR is set but no DINOv2 weights were found at: {local}"
        )

    bundled = bundled_model_dir()
    if _is_valid_local_model(bundled):
        return str(bundled), True

    offline = os.environ.get("TILEVISION_OFFLINE_MODEL", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if offline:
        raise FileNotFoundError(
            "TILEVISION_OFFLINE_MODEL is set but no bundled DINOv2 weights were found. "
            f"Run: python scripts/download_dinov2_model.py "
            f"(expected at {bundled})"
        )

    return DEFAULT_MODEL_ID, False


def model_status_message() -> str:
    """Human-readable summary for preflight / setup checks."""
    try:
        source, local_only = resolve_dinov2_model_source()
    except FileNotFoundError as exc:
        return f"Missing: {exc}"

    if local_only:
        return f"Ready (local weights at {source})"
    return f"Will download from Hugging Face on first use ({source})"
