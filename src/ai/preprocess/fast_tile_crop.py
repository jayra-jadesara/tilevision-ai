"""
Fast OpenCV tile-region isolation for room / scene query photos.

No neural network — typically a few milliseconds on CPU. Clean catalogue
tiles should skip this path; only scene-like queries benefit.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

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
    candidates = list_tile_region_candidates(image, limit=1)
    if candidates:
        return candidates[0]

    rgb = image.convert("RGB")
    width, height = rgb.size
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


def list_tile_region_candidates(
    image: Image.Image,
    *,
    limit: int = 3,
) -> list[TileCropResult]:
    """
    Return up to ``limit`` distinct tile-like crops, best first.

    Used for multi-crop query embedding (offline accuracy boost).
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 64 or height < 64:
        return [
            TileCropResult(
                image=rgb,
                box=(0, 0, width, height),
                confidence=0.0,
                method="none",
            )
        ]

    scale = min(1.0, _DETECT_MAX_EDGE / float(max(width, height)))
    work_w = max(1, int(round(width * scale)))
    work_h = max(1, int(round(height * scale)))
    work = np.asarray(rgb.resize((work_w, work_h), Image.Resampling.BILINEAR))

    scored: list[tuple[float, tuple[int, int, int, int], str]] = []

    for box in _tile_like_contours(work):
        scored.append((_score_box(box, work_w, work_h), box, "contour"))

    texture_box = _dominant_texture_box(work)
    if texture_box is not None:
        scored.append((_score_box(texture_box, work_w, work_h), texture_box, "texture"))

    floor_box = _lower_floor_band_box(work)
    if floor_box is not None:
        scored.append((_score_box(floor_box, work_w, work_h) + 0.04, floor_box, "floor_band"))

    scored.sort(key=lambda item: item[0], reverse=True)

    results: list[TileCropResult] = []
    used_boxes: list[tuple[int, int, int, int]] = []
    for score, box, method in scored:
        if score < _MIN_CONFIDENCE and method != "floor_band":
            continue
        if method == "floor_band" and score < 0.38:
            continue
        mapped = _map_box_to_source(box, scale=scale, src_w=width, src_h=height)
        mapped = _pad_box(*mapped, width, height, _PAD_RATIO)
        if any(_boxes_overlap(mapped, prev) > 0.72 for prev in used_boxes):
            continue
        used_boxes.append(mapped)
        left, top, right, bottom = mapped
        cropped = rgb.crop((left, top, right, bottom))
        results.append(
            TileCropResult(
                image=cropped,
                box=mapped,
                confidence=float(score),
                method=method,
            )
        )
        if len(results) >= max(1, int(limit)):
            break

    if not results:
        focus = _center_crop(rgb, ratio=0.70)
        fw, fh = focus.size
        left = (width - fw) // 2
        top = (height - fh) // 2
        results.append(
            TileCropResult(
                image=focus,
                box=(left, top, left + fw, top + fh),
                confidence=0.35,
                method="center_fallback",
            )
        )
    return results


def _boxes_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return inter / float(min(area_a, area_b))


def _tile_like_contours(
    work_rgb: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    height, width = gray.shape[:2]
    image_area = float(width * height)
    boxes: list[tuple[float, tuple[int, int, int, int]]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        area_ratio = area / image_area
        if area_ratio < _MIN_AREA_RATIO or area_ratio > _MAX_AREA_RATIO:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 24 or h < 24:
            continue

        aspect = w / max(h, 1)
        # Allow foreshortened floor tiles in room photos.
        if aspect < 0.35 or aspect > 2.8:
            continue

        rect_area = float(w * h)
        fill = area / max(rect_area, 1.0)
        if fill < 0.30:
            continue

        score = _score_box((x, y, x + w, y + h), width, height) + 0.05 * fill
        boxes.append((score, (x, y, x + w, y + h)))

    boxes.sort(key=lambda item: item[0], reverse=True)
    return [box for _score, box in boxes[:5]]


def _largest_tile_like_contour(
    work_rgb: np.ndarray,
) -> tuple[int, int, int, int] | None:
    boxes = _tile_like_contours(work_rgb)
    return boxes[0] if boxes else None


def _lower_floor_band_box(work_rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bias toward the lower two-thirds — common floor location in room shots."""
    height, width = work_rgb.shape[:2]
    top = int(height * 0.28)
    band = work_rgb[top:height, :]
    if band.size == 0:
        return None
    local = _dominant_texture_box(band)
    if local is None:
        # Center of the floor band.
        bw, bh = int(width * 0.55), int((height - top) * 0.70)
        left = (width - bw) // 2
        local_top = top + ((height - top) - bh) // 2
        return (left, local_top, left + bw, local_top + bh)
    x0, y0, x1, y1 = local
    return (x0, y0 + top, x1, y1 + top)


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
    aspect_score = 1.0 - min(abs(np.log(aspect)) / np.log(2.8), 1.0)

    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0
    dx = abs(cx - width / 2.0) / (width / 2.0)
    # Floor photos: slightly prefer lower half over upper furniture/ceiling.
    ideal_cy = height * 0.58
    dy = abs(cy - ideal_cy) / (height / 2.0)
    center_score = 1.0 - min(0.5 * (dx + dy), 1.0)

    return float(0.42 * size_score + 0.28 * aspect_score + 0.30 * center_score)


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


def _persist_last_crop(kind: str, image: Image.Image, temp_dir: Path) -> Path:
    """Stable copy for diagnosis after unique-named temp files disappear."""
    keep_path = temp_dir / f"last_{kind}.jpg"
    image.convert("RGB").save(str(keep_path), "JPEG", quality=95)
    return keep_path


def _already_full_frame_tile(image: Image.Image) -> bool:
    from src.ai.search_quality.query_analyzer import analyze_query, is_full_frame_tile

    return is_full_frame_tile(analyze_query(image))


def save_auto_tile_crop(image_path: str | Path) -> tuple[Path, TileCropResult]:
    """
    Isolate the tile region and write a JPEG under ``tilevision_crops``.

    Used by Search → Auto Crop & Search so the user can run DINOv2 on the
    cropped region explicitly (same temp-folder convention as manual crop).

    Clean close-ups skip floor_band isolation — that heuristic is for room
    photos and otherwise keeps ~30% of an already-framed tile.
    """
    path = Path(image_path)
    with Image.open(path) as img:
        source = ImageOps.exif_transpose(img.convert("RGB"))

    width, height = source.size
    if _already_full_frame_tile(source):
        result = TileCropResult(
            image=source,
            box=(0, 0, width, height),
            confidence=1.0,
            method="already_clean",
        )
        logger.info(
            "Auto crop skipped isolation for full-frame tile %s (%dx%d)",
            path.name,
            width,
            height,
        )
    else:
        result = isolate_tile_region(source)
    temp_dir = Path(tempfile.gettempdir()) / "tilevision_crops"
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_path = temp_dir / f"autocrop_{path.stem}_{id(result)}.jpg"
    result.image.convert("RGB").save(str(out_path), "JPEG", quality=95)
    keep_path = _persist_last_crop("autocrop", result.image, temp_dir)
    logger.info(
        "Saved auto tile crop: %s (method=%s conf=%.2f keep=%s)",
        out_path.name,
        result.method,
        result.confidence,
        keep_path.name,
    )
    return out_path, result
