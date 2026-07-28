"""
Fast OpenCV tile-region isolation for room / scene query photos.

No neural network — typically a few milliseconds on CPU. Clean catalogue
tiles should skip this path; only scene-like queries benefit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("tilevision.ai.fast_tile_crop")

# Working resolution for detection (keeps CPU cost low on Mac Intel).
_DETECT_MAX_EDGE = 640
_MIN_CONFIDENCE = 0.42
_MIN_AREA_RATIO = 0.08
_MAX_AREA_RATIO = 0.92
_PAD_RATIO = 0.04


@dataclass(frozen=True, slots=True)
class TileCropResult:
    """Outcome of fast tile isolation."""

    image: Image.Image
    box: tuple[int, int, int, int]  # left, top, right, bottom in source coords
    confidence: float
    method: str

    @property
    def applied(self) -> bool:
        return self.method in {"contour", "texture"} and self.confidence >= _MIN_CONFIDENCE


def isolate_tile_region(image: Image.Image) -> TileCropResult:
    """
    Crop to the most likely tile surface in a room/installation photo.

    Falls back to a conservative center crop when no confident region is found.
    Never expands beyond the source image.
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 64 or height < 64:
        return TileCropResult(image=rgb, box=(0, 0, width, height), confidence=0.0, method="none")

    scale = min(1.0, _DETECT_MAX_EDGE / float(max(width, height)))
    work_w = max(1, int(round(width * scale)))
    work_h = max(1, int(round(height * scale)))
    work = np.asarray(rgb.resize((work_w, work_h), Image.Resampling.BILINEAR))

    candidates: list[tuple[float, tuple[int, int, int, int], str]] = []

    contour_box = _largest_tile_like_contour(work)
    if contour_box is not None:
        score = _score_box(contour_box, work_w, work_h)
        candidates.append((score, contour_box, "contour"))

    texture_box = _dominant_texture_box(work)
    if texture_box is not None:
        score = _score_box(texture_box, work_w, work_h)
        candidates.append((score, texture_box, "texture"))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_box, method = candidates[0]
        if best_score >= _MIN_CONFIDENCE:
            left, top, right, bottom = _map_box_to_source(
                best_box, scale=scale, src_w=width, src_h=height
            )
            left, top, right, bottom = _pad_box(
                left, top, right, bottom, width, height, _PAD_RATIO
            )
            cropped = rgb.crop((left, top, right, bottom))
            logger.info(
                "Fast tile crop (%s): conf=%.2f box=(%d,%d,%d,%d) from %dx%d",
                method,
                best_score,
                left,
                top,
                right,
                bottom,
                width,
                height,
            )
            return TileCropResult(
                image=cropped,
                box=(left, top, right, bottom),
                confidence=float(best_score),
                method=method,
            )

    # Conservative fallback — better than searching the full cluttered frame.
    focus = _center_crop(rgb, ratio=0.70)
    fw, fh = focus.size
    left = (width - fw) // 2
    top = (height - fh) // 2
    logger.debug("Fast tile crop fallback to center focus on %dx%d scene", width, height)
    return TileCropResult(
        image=focus,
        box=(left, top, left + fw, top + fh),
        confidence=0.35,
        method="center_fallback",
    )


def _largest_tile_like_contour(
    work_rgb: np.ndarray,
) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    height, width = gray.shape[:2]
    image_area = float(width * height)
    best: tuple[float, tuple[int, int, int, int]] | None = None

    for contour in contours:
        area = float(cv2.contourArea(contour))
        area_ratio = area / image_area
        if area_ratio < _MIN_AREA_RATIO or area_ratio > _MAX_AREA_RATIO:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 24 or h < 24:
            continue

        aspect = w / max(h, 1)
        # Floor/wall tiles are usually near-square in the photo plane.
        if aspect < 0.45 or aspect > 2.2:
            continue

        rect_area = float(w * h)
        fill = area / max(rect_area, 1.0)
        if fill < 0.35:
            continue

        score = _score_box((x, y, x + w, y + h), width, height) + 0.05 * fill
        box = (x, y, x + w, y + h)
        if best is None or score > best[0]:
            best = (score, box)

    return None if best is None else best[1]


def _dominant_texture_box(work_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    sq = cv2.GaussianBlur(gray * gray, (15, 15), 0)
    variance = np.maximum(sq - blur * blur, 0.0)

    # Adaptive threshold: keep the top textured regions.
    thresh = max(18.0, float(np.percentile(variance, 70)))
    mask = (variance >= thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    coords = cv2.findNonZero(mask)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    height, width = gray.shape[:2]
    area_ratio = (w * h) / float(width * height)
    if area_ratio < _MIN_AREA_RATIO or area_ratio > _MAX_AREA_RATIO:
        return None
    return (x, y, x + w, y + h)


def _score_box(box: tuple[int, int, int, int], width: int, height: int) -> float:
    left, top, right, bottom = box
    w = max(1, right - left)
    h = max(1, bottom - top)
    area_ratio = (w * h) / float(max(width * height, 1))

    # Prefer mid-sized regions (tile surface), not tiny objects or full frame.
    if area_ratio < 0.12:
        size_score = area_ratio / 0.12
    elif area_ratio <= 0.65:
        size_score = 1.0
    else:
        size_score = max(0.0, 1.0 - (area_ratio - 0.65) / 0.35)

    aspect = w / float(h)
    aspect_score = 1.0 - min(abs(np.log(aspect)) / np.log(2.2), 1.0)

    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0
    dx = abs(cx - width / 2.0) / (width / 2.0)
    dy = abs(cy - height / 2.0) / (height / 2.0)
    center_score = 1.0 - min(0.5 * (dx + dy), 1.0)

    return float(0.45 * size_score + 0.30 * aspect_score + 0.25 * center_score)


def _map_box_to_source(
    box: tuple[int, int, int, int],
    *,
    scale: float,
    src_w: int,
    src_h: int,
) -> tuple[int, int, int, int]:
    if scale <= 0:
        scale = 1.0
    left = int(round(box[0] / scale))
    top = int(round(box[1] / scale))
    right = int(round(box[2] / scale))
    bottom = int(round(box[3] / scale))
    left = max(0, min(left, src_w - 1))
    top = max(0, min(top, src_h - 1))
    right = max(left + 1, min(right, src_w))
    bottom = max(top + 1, min(bottom, src_h))
    return left, top, right, bottom


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


def _center_crop(image: Image.Image, ratio: float = 0.70) -> Image.Image:
    width, height = image.size
    crop_w = max(1, int(width * ratio))
    crop_h = max(1, int(height * ratio))
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h))
