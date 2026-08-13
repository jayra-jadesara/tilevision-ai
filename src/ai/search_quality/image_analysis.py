"""
Lightweight OpenCV / NumPy heuristics for catalog image analysis.

Used to decide whether extra index views are beneficial. Never invents
blind quadrant crops — only proposes views when signals support them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image


class ImageKind(str, Enum):
    CLEAN_TILE = "clean_tile"
    CATALOG_SHEET = "catalog_sheet"
    BORDERED_TILE = "bordered_tile"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ImageAnalysis:
    kind: ImageKind
    width: int
    height: int
    aspect: float
    white_border_ratio: float
    texture_richness: float
    text_region_score: float
    has_preview_grid: bool
    left_panel_beneficial: bool
    center_crop_beneficial: bool
    quality_score: float


def _white_border_ratio(rgb: np.ndarray) -> float:
    """
    Fraction of border pixels that look like blank catalogue margins.

    Light ceramic faces must NOT count as borders — require near-uniform
    high-key strips (low local variance), not merely bright marble.
    """
    h, w = rgb.shape[:2]
    if h < 32 or w < 32:
        return 0.0
    mx, my = max(1, w // 12), max(1, h // 12)
    strips = [
        rgb[:my, :],
        rgb[h - my :, :],
        rgb[:, :mx],
        rgb[:, w - mx :],
    ]
    hits = 0
    total = 0
    for strip in strips:
        flat = strip.reshape(-1, 3).astype(np.float32)
        total += flat.shape[0]
        bright = flat.mean(axis=1) > 235
        # Per-pixel channel spread — true white margins are flat.
        chan_spread = flat.max(axis=1) - flat.min(axis=1)
        uniform = chan_spread < 12.0
        # Local neighborhood variance proxy: distance from strip mean.
        strip_mean = flat.mean(axis=0)
        dist = np.linalg.norm(flat - strip_mean, axis=1)
        near_mean = dist < 18.0
        hits += int(np.sum(bright & uniform & near_mean))
    return float(hits / max(1, total))


def _texture_richness(gray: np.ndarray) -> float:
    """Laplacian variance normalized to a soft 0–1 band."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())
    # Ceramic veins often land in 20–400; map logarithmically.
    return float(np.clip(np.log1p(var) / np.log1p(400.0), 0.0, 1.0))


def _text_region_score(gray: np.ndarray) -> float:
    """
    Approximate text/logo density in the marketing column (right of slab panel).

    Previous implementation averaged Canny edge density over the entire right
    45% at a single threshold (60, 140). That undercounted real showroom
    sheets (PGYS2319): sparse gold logo lettering and small Chinese captions
    sit in the top ~35% of the column, while most of the strip is white
    margin or photo-grid cells — diluting the mean to ~0.006 (score 0.024).

    Uses three complementary signals on the marketing column (from ~42% width):
    1. Multi-threshold Canny on the *top band* (sparse large logo strokes).
    2. Horizontal Sobel row activity (stacked text lines / product code).
    3. Full-column Canny (grid lines + dense captions), preserved for parity.
    """
    h, w = gray.shape
    if w < 80 or h < 80:
        return 0.0

    # Marketing column starts near the slab split (~42–45%), not 55%.
    col_start = int(w * 0.42)
    right = gray[:, col_start:]
    if right.size == 0:
        return 0.0

    top_h = max(32, int(h * 0.38))
    top_band = right[:top_h, :]
    left_top = gray[:top_h, : max(1, col_start)]

    scores: list[float] = []

    # Signal 1: multi-threshold Canny on top band — gold logo on white is soft.
    for lo, hi in ((25, 75), (40, 100), (60, 140)):
        edges = cv2.Canny(top_band, lo, hi)
        density = float(np.mean(edges > 0))
        scores.append(density * 8.0)

    # Signal 2: horizontal stroke activity (text rows, product code line).
    sobel_x = cv2.Sobel(top_band, cv2.CV_64F, 1, 0, ksize=3)
    row_activity = np.mean(np.abs(sobel_x), axis=1)
    if row_activity.size > 0:
        peak = float(np.percentile(row_activity, 80))
        active_rows = float(np.mean(row_activity > max(peak * 0.45, 8.0)))
        scores.append(active_rows * 3.0)

    # Signal 3: legacy full-column edge density (grid dividers + captions).
    edges_full = cv2.Canny(right, 60, 140)
    scores.append(float(np.mean(edges_full > 0)) * 4.0)

    raw = float(np.clip(max(scores), 0.0, 1.0))

    # Suppress false positives on uniform tile faces: the marketing top band
    # must show more structured edges than the slab top band next to it.
    if left_top.size > 0:
        marketing_edges = cv2.Canny(top_band, 40, 100)
        slab_edges = cv2.Canny(left_top, 40, 100)
        m_density = float(np.mean(marketing_edges > 0))
        s_density = float(np.mean(slab_edges > 0))
        if m_density < s_density + 0.006:
            raw *= 0.30

    return float(np.clip(raw, 0.0, 1.0))


def _has_preview_grid(gray: np.ndarray) -> bool:
    h, w = gray.shape
    if w < 200 or h < 200:
        return False
    right = gray[:, int(w * 0.52) :]
    edges = cv2.Canny(right, 50, 120)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=8
    )
    if lines is None:
        return False
    return len(lines) >= 8


# Real PGYS2319 is aspect ~1.063 with a preview grid — the old 1.12 gate
# blocked panel isolation even when has_preview_grid=True.
_MARKETING_SHEET_MIN_ASPECT = 1.03
_DEFAULT_SHEET_MIN_ASPECT = 1.12


def _min_panel_aspect(
    aspect: float,
    *,
    has_preview_grid: bool,
    text_region_score: float,
) -> float:
    """Minimum aspect for left-panel isolation on marketing sheets."""
    if has_preview_grid or text_region_score >= 0.12:
        return _MARKETING_SHEET_MIN_ASPECT
    return _DEFAULT_SHEET_MIN_ASPECT


def marketing_sheet_panel_eligible(
    width: int,
    height: int,
    gray: np.ndarray,
) -> bool:
    """
    Whether ``primary_texture_panel`` should run (mirrors analyze_image gates).

    Exported for index-time cropping without duplicating thresholds.
    """
    if width < 480 or height < 320:
        return False
    aspect = width / max(height, 1)
    if aspect < _MARKETING_SHEET_MIN_ASPECT:
        return False
    if aspect >= _DEFAULT_SHEET_MIN_ASPECT:
        return True
    grid = _has_preview_grid(gray)
    text_score = _text_region_score(gray)
    return aspect >= _min_panel_aspect(
        aspect,
        has_preview_grid=grid,
        text_region_score=text_score,
    )


def analyze_image(image: Image.Image) -> ImageAnalysis:
    """Analyze a catalog image and recommend whether aux views help."""
    rgb_img = image.convert("RGB")
    width, height = rgb_img.size
    aspect = width / max(height, 1)
    rgb = np.asarray(rgb_img, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    border = _white_border_ratio(rgb)
    texture = _texture_richness(gray)
    text_score = _text_region_score(gray)
    grid = _has_preview_grid(gray)

    min_aspect = _min_panel_aspect(
        aspect,
        has_preview_grid=grid,
        text_region_score=text_score,
    )
    has_marketing_column = text_score >= 0.12 or grid or border >= 0.25
    left_panel = (
        aspect >= min_aspect
        and width >= 480
        and height >= 320
        and has_marketing_column
    )
    # Center crop helps large clean tiles when texture is present and the
    # frame is not already a tight crop. Block on wide preview-grid sheets
    # (center includes photo-grid cells — PGYS2319 failure mode).
    center = (
        min(width, height) >= 400
        and texture >= 0.15
        and not left_panel
        and not (grid and aspect >= _MARKETING_SHEET_MIN_ASPECT)
        and border < 0.55
    )

    if left_panel:
        kind = ImageKind.CATALOG_SHEET
    elif border >= 0.35:
        kind = ImageKind.BORDERED_TILE
    elif texture >= 0.12 and 0.75 <= aspect <= 1.35:
        kind = ImageKind.CLEAN_TILE
    else:
        kind = ImageKind.UNKNOWN

    quality = float(
        np.clip(
            0.35 * texture + 0.25 * (1.0 - min(border, 1.0)) + 0.20
            + (0.20 if kind != ImageKind.UNKNOWN else 0.0),
            0.0,
            1.0,
        )
    )

    return ImageAnalysis(
        kind=kind,
        width=width,
        height=height,
        aspect=float(aspect),
        white_border_ratio=border,
        texture_richness=texture,
        text_region_score=text_score,
        has_preview_grid=grid,
        left_panel_beneficial=left_panel,
        center_crop_beneficial=center,
        quality_score=quality,
    )
