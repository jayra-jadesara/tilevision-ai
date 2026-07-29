"""
Precise tile isolation for room photos (experimental).

Order:
  1. Fast OpenCV proposes a seed box
  2. SAM 2 refines the mask when TILEVISION_ENABLE_SAM2=1 and deps/weights exist
  3. Otherwise GrabCut refines the seed box (always available, still CPU-fast)

Default search / Auto Crop paths do NOT call this module.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.ai.preprocess.fast_tile_crop import isolate_tile_region

logger = logging.getLogger("tilevision.ai.precise_tile_crop")

_PAD_RATIO = 0.03
_MIN_MASK_AREA_RATIO = 0.05


@dataclass(frozen=True, slots=True)
class PreciseCropResult:
    image: Image.Image
    box: tuple[int, int, int, int]
    confidence: float
    method: str  # "sam2" | "grabcut" | "fast_fallback"
    detail: str = ""


def precise_isolate_tile(image: Image.Image) -> PreciseCropResult:
    """Isolate the tile surface with the best available precise backend."""
    rgb = ImageOps.exif_transpose(image.convert("RGB"))
    seed = isolate_tile_region(rgb)
    seed_box = seed.box

    # Try experimental SAM2 first when enabled.
    try:
        from src.ai.preprocess import sam2_backend

        if sam2_backend.sam2_enabled_by_env() and sam2_backend.sam2_api_available():
            mask = sam2_backend.segment_tile_mask(rgb, box=seed_box)
            cropped, box = _crop_from_mask(rgb, mask)
            if cropped is not None and box is not None:
                logger.info("Precise crop via SAM2 (%s)", sam2_backend.sam2_status())
                return PreciseCropResult(
                    image=cropped,
                    box=box,
                    confidence=0.85,
                    method="sam2",
                    detail=sam2_backend.sam2_status(),
                )
    except Exception as exc:
        logger.warning("SAM2 precise crop unavailable — using GrabCut. (%s)", exc)

    grab = _grabcut_refine(rgb, seed_box)
    if grab is not None:
        cropped, box, conf = grab
        return PreciseCropResult(
            image=cropped,
            box=box,
            confidence=conf,
            method="grabcut",
            detail="OpenCV GrabCut refine (SAM2 not active)",
        )

    # Last resort: the fast seed crop itself.
    return PreciseCropResult(
        image=seed.image,
        box=seed.box,
        confidence=max(seed.confidence, 0.35),
        method="fast_fallback",
        detail=f"fast:{seed.method}",
    )


def save_precise_tile_crop(image_path: str | Path) -> tuple[Path, PreciseCropResult]:
    """Write a precise crop JPEG under ``tilevision_crops`` and return its path."""
    path = Path(image_path)
    with Image.open(path) as img:
        source = ImageOps.exif_transpose(img.convert("RGB"))

    result = precise_isolate_tile(source)
    temp_dir = Path(tempfile.gettempdir()) / "tilevision_crops"
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_path = temp_dir / f"precise_{path.stem}_{id(result)}.jpg"
    result.image.convert("RGB").save(str(out_path), "JPEG", quality=92)
    logger.info(
        "Saved precise tile crop: %s (method=%s conf=%.2f)",
        out_path.name,
        result.method,
        result.confidence,
    )
    return out_path, result


def _crop_from_mask(
    image: Image.Image,
    mask: np.ndarray,
) -> tuple[Image.Image, tuple[int, int, int, int]] | tuple[None, None]:
    height, width = mask.shape[:2]
    if mask.shape != (image.size[1], image.size[0]):
        return None, None

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None, None

    area_ratio = float(len(xs)) / float(max(width * height, 1))
    if area_ratio < _MIN_MASK_AREA_RATIO:
        return None, None

    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    left, top, right, bottom = _pad_box(left, top, right, bottom, width, height, _PAD_RATIO)
    return image.crop((left, top, right, bottom)), (left, top, right, bottom)


def _grabcut_refine(
    image: Image.Image,
    seed_box: tuple[int, int, int, int],
) -> tuple[Image.Image, tuple[int, int, int, int], float] | None:
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    left, top, right, bottom = seed_box
    left = max(0, min(left, width - 2))
    top = max(0, min(top, height - 2))
    right = max(left + 2, min(right, width))
    bottom = max(top + 2, min(bottom, height))
    rect = (left, top, right - left, bottom - top)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros((height, width), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except Exception as exc:
        logger.debug("GrabCut failed: %s", exc)
        return None

    foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    if foreground.sum() < (width * height) * _MIN_MASK_AREA_RATIO:
        # Fall back to the seed rectangle itself.
        cropped = image.crop((left, top, right, bottom))
        return cropped, (left, top, right, bottom), 0.55

    ys, xs = np.where(foreground > 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, y0, x1, y1 = _pad_box(x0, y0, x1, y1, width, height, _PAD_RATIO)
    cropped = image.crop((x0, y0, x1, y1))
    return cropped, (x0, y0, x1, y1), 0.70


def _pad_box(
    left: int,
    top: int,
    right: int,
    bottom: int,
    width: int,
    height: int,
    pad_ratio: float,
) -> tuple[int, int, int, int]:
    bw = right - left
    bh = bottom - top
    pad_x = int(round(bw * pad_ratio))
    pad_y = int(round(bh * pad_ratio))
    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(width, right + pad_x)
    bottom = min(height, bottom + pad_y)
    return left, top, right, bottom
