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
    Approximate text/logo density via MSER-like contrast blobs on one side.

    High when a vertical strip has many small high-contrast components
    (typical catalogue info column).
    """
    h, w = gray.shape
    if w < 80 or h < 80:
        return 0.0
    right = gray[:, int(w * 0.55) :]
    edges = cv2.Canny(right, 60, 140)
    density = float(np.mean(edges > 0))
    return float(np.clip(density * 4.0, 0.0, 1.0))


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

    left_panel = (
        aspect >= 1.12
        and width >= 480
        and height >= 320
        and (text_score >= 0.12 or grid or border >= 0.25)
    )
    # Center crop helps large clean tiles when texture is present and the
    # frame is not already a tight crop.
    center = (
        min(width, height) >= 400
        and texture >= 0.15
        and not left_panel
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
