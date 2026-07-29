"""
Perspective straighten for angled floor / wall tile photos (query-only).

Uses classical OpenCV — no neural nets, fully offline. On failure returns
the original image unchanged.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("tilevision.ai.perspective_straighten")

_MAX_EDGE = 640
_MIN_AREA_RATIO = 0.18
_MAX_AREA_RATIO = 0.98


def straighten_tile_view(image: Image.Image) -> Image.Image:
    """
    Warp the dominant near-rectangular region to a frontal view when possible.

    Safe no-op when no reliable quad is found (catalogue tiles, soft textures).
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 80 or height < 80:
        return rgb

    scale = min(1.0, _MAX_EDGE / float(max(width, height)))
    work_w = max(1, int(round(width * scale)))
    work_h = max(1, int(round(height * scale)))
    work = np.asarray(rgb.resize((work_w, work_h), Image.Resampling.BILINEAR))

    quad = _find_dominant_quad(work)
    if quad is None:
        return rgb

    # Map quad back to full-resolution coordinates.
    src = (quad.astype(np.float32) / max(scale, 1e-6)).astype(np.float32)
    src = _order_quad_points(src)

    side = int(
        max(
            np.linalg.norm(src[0] - src[1]),
            np.linalg.norm(src[1] - src[2]),
            np.linalg.norm(src[2] - src[3]),
            np.linalg.norm(src[3] - src[0]),
            64.0,
        )
    )
    side = int(min(max(side, 128), max(width, height) * 2, 1600))
    dst = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
        dtype=np.float32,
    )

    try:
        matrix = cv2.getPerspectiveTransform(src, dst)
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        warped = cv2.warpPerspective(
            bgr,
            matrix,
            (side, side),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        out = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        logger.info(
            "Perspective straighten applied: %dx%d → %dx%d",
            width,
            height,
            out.size[0],
            out.size[1],
        )
        return out
    except Exception as exc:
        logger.debug("Perspective straighten skipped: %s", exc)
        return rgb


def _find_dominant_quad(work_rgb: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    height, width = gray.shape[:2]
    image_area = float(width * height)
    best: tuple[float, np.ndarray] | None = None

    for contour in contours:
        area = float(cv2.contourArea(contour))
        ratio = area / image_area
        if ratio < _MIN_AREA_RATIO or ratio > _MAX_AREA_RATIO:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        pts = approx.reshape(4, 2).astype(np.float32)
        # Reject near-degenerate quads.
        if _min_side_length(pts) < 24:
            continue

        rect = cv2.minAreaRect(approx)
        (_, _), (rw, rh), _angle = rect
        if rw < 1 or rh < 1:
            continue
        aspect = max(rw, rh) / max(min(rw, rh), 1.0)
        if aspect > 3.2:
            continue

        score = ratio * (1.0 / (1.0 + abs(aspect - 1.0)))
        if best is None or score > best[0]:
            best = (score, pts)

    return None if best is None else best[1]


def _min_side_length(pts: np.ndarray) -> float:
    ordered = _order_quad_points(pts)
    lengths = [
        float(np.linalg.norm(ordered[i] - ordered[(i + 1) % 4]))
        for i in range(4)
    ]
    return min(lengths)


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Return points in TL, TR, BR, BL order."""
    points = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    tl = points[np.argmin(sums)]
    br = points[np.argmax(sums)]
    tr = points[np.argmin(diffs)]
    bl = points[np.argmax(diffs)]
    return np.array([tl, tr, br, bl], dtype=np.float32)
