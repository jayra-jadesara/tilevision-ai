"""
Precise tile isolation for room photos (experimental).

Works on **Windows, Mac Intel, and Mac Apple Silicon** with the same path:

  1. Fast OpenCV proposes a seed box
  2. SAM 2 refines when ``TILEVISION_ENABLE_SAM2=1`` and the local stack can load it
     (Windows / Mac Intel / Mac Silicon — no OS blacklist)
  3. Otherwise OpenCV GrabCut refines the seed (universal CPU fallback)
  4. Fast seed crop is the last resort — never fails the button

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
_GRABCUT_MAX_EDGE = 640
_GRABCUT_ITERS = 4


@dataclass(frozen=True, slots=True)
class PreciseCropResult:
    image: Image.Image
    box: tuple[int, int, int, int]
    confidence: float
    method: str  # "sam2" | "grabcut" | "fast_fallback"
    detail: str = ""


def expected_precise_backend() -> str:
    """
    Which backend Precise Crop will attempt on this machine.

    Returns ``sam2`` or ``grabcut``. GrabCut is the universal guarantee for
    Windows + Mac Intel + Mac Silicon when SAM2 is off or unsupported.
    """
    try:
        from src.ai.preprocess import sam2_backend

        if sam2_backend.sam2_should_run():
            return "sam2"
    except Exception:
        pass
    return "grabcut"


def precise_isolate_tile(image: Image.Image) -> PreciseCropResult:
    """Isolate the tile surface with the best available precise backend."""
    rgb = ImageOps.exif_transpose(image.convert("RGB"))
    seed = isolate_tile_region(rgb)
    seed_box = seed.box

    # Optional SAM2 — Windows / Mac Intel / Mac Silicon when stack allows.
    try:
        from src.ai.preprocess import sam2_backend

        if sam2_backend.sam2_should_run():
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
        detail = _grabcut_detail()
        return PreciseCropResult(
            image=cropped,
            box=box,
            confidence=conf,
            method="grabcut",
            detail=detail,
        )

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
        "Saved precise tile crop: %s (method=%s conf=%.2f detail=%s)",
        out_path.name,
        result.method,
        result.confidence,
        result.detail,
    )
    return out_path, result


def _grabcut_detail() -> str:
    from src.utils.platform_info import is_mac_intel, is_macos, is_windows

    if is_mac_intel():
        return "GrabCut on Mac Intel (SAM2 not loaded — install experimental stack to enable)"
    if is_macos():
        return "GrabCut on Mac Apple Silicon (SAM2 not active)"
    if is_windows():
        return "GrabCut on Windows (SAM2 not active)"
    return "GrabCut (SAM2 not active)"


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
    """
    Refine the seed rectangle with GrabCut.

    Runs on a downscaled working image for speed on Mac Intel / large phone
    photos, then maps the result box back to full resolution.
    """
    width, height = image.size
    left, top, right, bottom = seed_box
    left = max(0, min(left, width - 2))
    top = max(0, min(top, height - 2))
    right = max(left + 2, min(right, width))
    bottom = max(top + 2, min(bottom, height))

    scale = min(1.0, _GRABCUT_MAX_EDGE / float(max(width, height)))
    work_w = max(2, int(round(width * scale)))
    work_h = max(2, int(round(height * scale)))

    work_rgb = image.resize((work_w, work_h), Image.Resampling.BILINEAR)
    rgb = np.asarray(work_rgb.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    wl = max(0, min(int(round(left * scale)), work_w - 2))
    wt = max(0, min(int(round(top * scale)), work_h - 2))
    wr = max(wl + 2, min(int(round(right * scale)), work_w))
    wb = max(wt + 2, min(int(round(bottom * scale)), work_h))
    rect = (wl, wt, wr - wl, wb - wt)

    mask = np.zeros((work_h, work_w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, _GRABCUT_ITERS, cv2.GC_INIT_WITH_RECT)
    except Exception as exc:
        logger.debug("GrabCut failed: %s", exc)
        cropped = image.crop((left, top, right, bottom))
        return cropped, (left, top, right, bottom), 0.50

    foreground = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
    ).astype(np.uint8)
    min_pixels = max(32, int(work_w * work_h * _MIN_MASK_AREA_RATIO))
    if int(foreground.sum()) < min_pixels:
        cropped = image.crop((left, top, right, bottom))
        return cropped, (left, top, right, bottom), 0.55

    ys, xs = np.where(foreground > 0)
    x0 = int(round(int(xs.min()) / scale))
    x1 = int(round((int(xs.max()) + 1) / scale))
    y0 = int(round(int(ys.min()) / scale))
    y1 = int(round((int(ys.max()) + 1) / scale))
    x0, y0, x1, y1 = _pad_box(x0, y0, x1, y1, width, height, _PAD_RATIO)
    # Keep result from collapsing / exploding.
    if (x1 - x0) < width * 0.08 or (y1 - y0) < height * 0.08:
        cropped = image.crop((left, top, right, bottom))
        return cropped, (left, top, right, bottom), 0.55
    if (x1 - x0) * (y1 - y0) > width * height * 0.95:
        cropped = image.crop((left, top, right, bottom))
        return cropped, (left, top, right, bottom), 0.55

    cropped = image.crop((x0, y0, x1, y1))
    return cropped, (x0, y0, x1, y1), 0.72


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
