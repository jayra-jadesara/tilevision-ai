"""
Optional SAM 2 backend for precise tile segmentation (experimental).

Not used by default search. Same enable path on:
  - Windows
  - Mac Intel
  - Mac Apple Silicon
  - Linux

Requires transformers with Sam2Model (typically 5.x) + capable torch.
If SAM2 cannot load on a machine, Precise Crop falls back to GrabCut.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger("tilevision.ai.sam2_backend")

DEFAULT_SAM2_MODEL_ID = "facebook/sam2.1-hiera-tiny"
_BUNDLED_DIRNAME = "sam2.1-hiera-tiny"
_MIN_TORCH = (2, 5, 0)

_model: Any = None
_processor: Any = None
_load_error: str | None = None
# Set from AppSettings so the UI toggle works without restarting env vars.
_settings_enabled: bool = False


def configure_sam2_from_settings(enabled: bool) -> None:
    """Apply the Settings → Experimental SAM2 checkbox at runtime."""
    global _settings_enabled
    _settings_enabled = bool(enabled)
    logger.info("SAM2 precise-crop setting %s", "ON" if _settings_enabled else "OFF")


def sam2_enabled_by_env() -> bool:
    """Feature flag from environment (lab / CI)."""
    value = os.environ.get("TILEVISION_ENABLE_SAM2", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def sam2_enabled_by_settings() -> bool:
    """Feature flag from persisted AppSettings."""
    return bool(_settings_enabled)


def sam2_enabled() -> bool:
    """True when either Settings or env enables experimental SAM2."""
    return sam2_enabled_by_settings() or sam2_enabled_by_env()


def sam2_api_available() -> bool:
    """True when the installed transformers build exports Sam2 classes."""
    try:
        from transformers import Sam2Model, Sam2Processor  # noqa: F401

        return True
    except Exception:
        return False


def _torch_version_tuple() -> tuple[int, int, int]:
    try:
        import torch

        match = re.match(r"(\d+)\.(\d+)\.(\d+)", torch.__version__.split("+")[0])
        if not match:
            return (0, 0, 0)
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except Exception:
        return (0, 0, 0)


def sam2_platform_supported() -> bool:
    """
    Return True when this machine's Python stack can load SAM2.

    No OS blacklist — Windows, Mac Intel, and Mac Silicon all qualify when
    Sam2Model exists and torch is new enough (or TILEVISION_SAM2_FORCE=1).
    Older stacks simply report False and Precise Crop uses GrabCut.
    """
    if not sam2_api_available():
        return False
    if os.environ.get("TILEVISION_SAM2_FORCE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True
    return _torch_version_tuple() >= _MIN_TORCH


def sam2_should_run() -> bool:
    """True when Settings/env enabled SAM2 and this machine can run it."""
    return sam2_enabled() and sam2_platform_supported()


def resolve_sam2_model_source() -> tuple[str, bool]:
    """
    Return (model_source, local_files_only).

    Looks for:
      1. TILEVISION_SAM2_MODEL_DIR
      2. model_weights/sam2.1-hiera-tiny/
      3. Hugging Face id (download when online)
    """
    env_dir = os.environ.get("TILEVISION_SAM2_MODEL_DIR", "").strip()
    if env_dir:
        local = Path(env_dir).expanduser()
        if local.is_dir():
            return str(local), True
        raise FileNotFoundError(f"TILEVISION_SAM2_MODEL_DIR not found: {local}")

    from src.ai.model_paths import runtime_root

    bundled = runtime_root() / "model_weights" / _BUNDLED_DIRNAME
    if bundled.is_dir() and (bundled / "config.json").is_file():
        return str(bundled), True

    offline = os.environ.get("TILEVISION_OFFLINE_MODEL", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if offline:
        raise FileNotFoundError(
            "TILEVISION_OFFLINE_MODEL is set but no local SAM2 weights were found. "
            f"Place weights at {bundled} or set TILEVISION_SAM2_MODEL_DIR."
        )
    return DEFAULT_SAM2_MODEL_ID, False


def _platform_label() -> str:
    from src.utils.platform_info import is_mac_intel, is_macos, is_windows

    if is_windows():
        return "Windows"
    if is_mac_intel():
        return "Mac Intel"
    if is_macos():
        return "Mac Apple Silicon"
    return "Linux"


def sam2_status() -> str:
    if not sam2_enabled():
        return (
            f"Disabled on {_platform_label()} "
            "(enable in Settings → Experimental, or set TILEVISION_ENABLE_SAM2=1)"
        )
    if not sam2_api_available():
        return (
            f"Unavailable on {_platform_label()} "
            "(needs transformers with Sam2Model — see requirements-sam2-experimental.txt); "
            "Precise Crop uses GrabCut"
        )
    if _torch_version_tuple() < _MIN_TORCH and not os.environ.get(
        "TILEVISION_SAM2_FORCE", ""
    ).strip():
        return (
            f"Unavailable on {_platform_label()} "
            "(needs torch>=2.5.1 for SAM2, or TILEVISION_SAM2_FORCE=1); "
            "Precise Crop uses GrabCut"
        )
    if _load_error:
        return f"Load failed on {_platform_label()}: {_load_error}"
    if _model is not None:
        return f"Ready (loaded on {_platform_label()})"
    try:
        source, local_only = resolve_sam2_model_source()
    except FileNotFoundError as exc:
        return f"Missing weights on {_platform_label()}: {exc}"
    mode = "local" if local_only else "hub"
    return f"Enabled on {_platform_label()} ({mode}: {source})"


def _resolve_device():
    import torch

    from src.ai.gpu_info import configure_mps_fallback, detect_gpu_runtime

    configure_mps_fallback()
    info = detect_gpu_runtime(preference="auto")
    return torch.device(info.active_device)


def load_sam2_model() -> tuple[Any, Any]:
    """Lazy-load SAM2. Raises if unavailable or disabled on this platform."""
    global _model, _processor, _load_error

    if not sam2_enabled():
        raise RuntimeError(
            "SAM2 is disabled. Enable it in Settings → Experimental "
            "or set TILEVISION_ENABLE_SAM2=1."
        )
    if not sam2_platform_supported():
        raise RuntimeError(
            f"SAM2 is not available on {_platform_label()} with this Python stack. "
            "Precise Crop will use GrabCut instead."
        )
    if _model is not None and _processor is not None:
        return _model, _processor

    from transformers import Sam2Model, Sam2Processor

    source, local_only = resolve_sam2_model_source()
    device = _resolve_device()
    logger.info(
        "Loading experimental SAM2 from %s (local_only=%s) on %s (%s)",
        source,
        local_only,
        device,
        _platform_label(),
    )

    try:
        _processor = Sam2Processor.from_pretrained(source, local_files_only=local_only)
        _model = Sam2Model.from_pretrained(source, local_files_only=local_only)
        _model.to(device)
        _model.eval()
        _load_error = None
    except Exception as exc:
        _model = None
        _processor = None
        _load_error = str(exc)
        raise

    return _model, _processor


def segment_tile_mask(
    image: Image.Image,
    *,
    box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """
    Return a boolean HxW mask for the likely tile region.

    Uses a center point (and optional box) prompt derived from fast OpenCV crop.
    """
    import torch

    model, processor = load_sam2_model()
    rgb = image.convert("RGB")
    width, height = rgb.size

    if box is None:
        # Prompt the image center.
        cx, cy = width // 2, height // 2
        input_points = [[[[cx, cy]]]]
        input_boxes = None
    else:
        left, top, right, bottom = box
        cx = (left + right) // 2
        cy = (top + bottom) // 2
        input_points = [[[[cx, cy]]]]
        input_boxes = [[[float(left), float(top), float(right), float(bottom)]]]

    input_labels = [[[1]]]
    kwargs = {
        "images": rgb,
        "input_points": input_points,
        "input_labels": input_labels,
        "return_tensors": "pt",
    }
    if input_boxes is not None:
        kwargs["input_boxes"] = input_boxes

    inputs = processor(**kwargs)
    device = next(model.parameters()).device
    inputs = {
        key: (value.to(device) if hasattr(value, "to") else value)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"],
    )[0]
    # masks: (1, num_masks, H, W) or similar — pick highest-score mask when available.
    mask_tensor = masks
    if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
        scores = outputs.iou_scores.cpu().reshape(-1)
        best = int(scores.argmax().item())
        binary = mask_tensor[0, best].numpy() > 0.5
    else:
        # Fallback: first mask or union of masks.
        arr = mask_tensor[0].numpy()
        if arr.ndim == 3:
            binary = arr.max(axis=0) > 0.5
        else:
            binary = arr > 0.5

    if binary.shape[0] != height or binary.shape[1] != width:
        # Resize mask to original image size if processor returned scaled mask.
        mask_img = Image.fromarray((binary.astype(np.uint8) * 255))
        mask_img = mask_img.resize((width, height), Image.Resampling.NEAREST)
        binary = np.asarray(mask_img) > 127

    return binary.astype(bool)
