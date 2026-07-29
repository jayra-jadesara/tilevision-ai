"""
Optional SAM 2 backend for precise tile segmentation (experimental).

Not used by default search. Requires newer transformers with Sam2Model
(typically transformers 5.x + recent torch). Mac Intel production pins
stay on transformers 4.x — this backend simply reports unavailable there.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger("tilevision.ai.sam2_backend")

DEFAULT_SAM2_MODEL_ID = "facebook/sam2.1-hiera-tiny"
_BUNDLED_DIRNAME = "sam2.1-hiera-tiny"

_model: Any = None
_processor: Any = None
_load_error: str | None = None


def sam2_enabled_by_env() -> bool:
    """Feature flag — off unless explicitly enabled for experimental use."""
    value = os.environ.get("TILEVISION_ENABLE_SAM2", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def sam2_api_available() -> bool:
    """True when the installed transformers build exports Sam2 classes."""
    try:
        from transformers import Sam2Model, Sam2Processor  # noqa: F401

        return True
    except Exception:
        return False


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


def sam2_status() -> str:
    if not sam2_enabled_by_env():
        return "Disabled (set TILEVISION_ENABLE_SAM2=1 to experiment)"
    if not sam2_api_available():
        return "Unavailable (needs transformers with Sam2Model — see requirements-sam2-experimental.txt)"
    if _load_error:
        return f"Load failed: {_load_error}"
    if _model is not None:
        return "Ready (loaded)"
    try:
        source, local_only = resolve_sam2_model_source()
    except FileNotFoundError as exc:
        return f"Missing weights: {exc}"
    mode = "local" if local_only else "hub"
    return f"Enabled ({mode}: {source})"


def _resolve_device():
    import torch

    from src.ai.gpu_info import configure_mps_fallback, detect_gpu_runtime

    configure_mps_fallback()
    info = detect_gpu_runtime(preference="auto")
    # Prefer CUDA/MPS when present; CPU is fine for tiny but slower.
    return torch.device(info.active_device)


def load_sam2_model() -> tuple[Any, Any]:
    """Lazy-load SAM2. Raises if unavailable or disabled."""
    global _model, _processor, _load_error

    if not sam2_enabled_by_env():
        raise RuntimeError("SAM2 is disabled. Set TILEVISION_ENABLE_SAM2=1.")
    if not sam2_api_available():
        raise RuntimeError(
            "Sam2Model is not available in this transformers build. "
            "Install requirements-sam2-experimental.txt on a supported machine."
        )
    if _model is not None and _processor is not None:
        return _model, _processor

    import torch
    from transformers import Sam2Model, Sam2Processor

    source, local_only = resolve_sam2_model_source()
    device = _resolve_device()
    logger.info("Loading experimental SAM2 from %s (local_only=%s) on %s", source, local_only, device)

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
