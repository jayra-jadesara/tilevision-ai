"""
Lightweight query-image analyzer (OpenCV / PIL / NumPy only).

Distinguishes clean tiles, catalogue sheets, room scenes, phone screenshots,
and partial crops so the QUERY pipeline can choose preprocessing. Does not
change the indexed catalog.

Root-cause note (v1.2.32 study): room photos false-triggered
``primary_texture_panel`` (wide aspect + textured left third) and were treated
as catalogue sheets, skipping isolation. Catalogue detection here requires
white-margin / text-column evidence — not grid Hough hits alone (rooms also
produce line segments).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image

from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.search_quality.image_analysis import analyze_image


class QueryKind(str, Enum):
    CLEAN_TILE = "clean_tile"
    CATALOG_SHEET = "catalog_sheet"
    ROOM_SCENE = "room_scene"
    PHONE_SCREENSHOT = "phone_screenshot"
    PARTIAL_CROP = "partial_crop"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    kind: QueryKind
    width: int
    height: int
    aspect_ratio: float
    tile_coverage_ratio: float
    background_ratio: float
    entropy: float
    gradient_density: float
    texture_density: float
    edge_density: float
    white_border_ratio: float
    largest_texture_area_ratio: float
    has_ui_chrome: bool
    text_region_score: float
    has_preview_grid: bool
    band_color_delta: float
    confidence: float


def _entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).ravel()
    p = hist / max(1.0, float(hist.sum()))
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / 6.0)


def _gradient_density(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return float(np.clip(np.mean(mag > 18.0), 0.0, 1.0))


def _edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 60, 140)
    return float(np.mean(edges > 0))


def _band_color_delta(rgb: np.ndarray) -> float:
    """Ceiling vs floor mean-color distance — high for room scenes."""
    h, w = rgb.shape[:2]
    if h < 40 or w < 40:
        return 0.0
    top = rgb[: h // 5].reshape(-1, 3).mean(axis=0)
    bot = rgb[4 * h // 5 :].reshape(-1, 3).mean(axis=0)
    return float(np.linalg.norm(top.astype(np.float32) - bot.astype(np.float32)))


def _largest_texture_region_ratio(gray: np.ndarray) -> tuple[float, float]:
    h, w = gray.shape
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mask = (mag > 12.0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    coverage = float(np.mean(mask > 0))
    if n <= 1:
        return 0.0, coverage
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = float(areas.max()) if len(areas) else 0.0
    return largest / float(max(h * w, 1)), coverage


def _has_ui_chrome(rgb: np.ndarray) -> bool:
    h, w = rgb.shape[:2]
    if h < 200 or w < 120:
        return False
    top_h = max(8, h // 16)
    top = rgb[:top_h, :]
    top_dark = float(top.mean()) < 45.0
    below = rgb[top_h : top_h + max(8, h // 20), :]
    contrast = float(below.mean()) - float(top.mean())
    aspect = w / max(h, 1)
    portrait_phone = 0.40 <= aspect <= 0.75
    return bool(top_dark and contrast > 25.0 and (portrait_phone or h >= 900))


def _is_catalog_sheet(
    *,
    aspect: float,
    width: int,
    height: int,
    white_border: float,
    text_score: float,
    band_delta: float,
    has_preview_grid: bool,
) -> bool:
    """
    True catalogue marketing sheets — NOT wide room photos and NOT
    rotation/brightness frames that merely inflate white_border_ratio.

    Measured (320-tile study):
      sheet: border≈0.41, text≈0.11, band_delta≈10, aspect≈1.33
      room:  border≈0.00, text≈0.03, band_delta≈98
      rotation expand-fill: border≈0.76 but aspect≈1.0 (not a sheet)
    """
    if width < 480 or height < 320 or aspect < 1.12:
        return False
    if band_delta >= 40.0:
        return False
    # Prefer text/grid + white margin. White margins alone are not enough —
    # bright ceramic faces and rotate-expand fill inflate border on squares
    # (aspect gate already excludes those).
    if text_score >= 0.10 and white_border >= 0.08:
        return True
    if has_preview_grid and white_border >= 0.20:
        return True
    if white_border >= 0.30 and text_score >= 0.08:
        return True
    return False


def analyze_query(image: Image.Image) -> QueryAnalysis:
    """Classify a search query and estimate geometric / texture signals."""
    rgb_img = image.convert("RGB")
    width, height = rgb_img.size
    aspect = width / max(height, 1)
    rgb = np.asarray(rgb_img, dtype=np.uint8)

    scale = min(1.0, 640.0 / float(max(width, height)))
    if scale < 1.0:
        small = cv2.resize(
            rgb,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = rgb
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    catalog = analyze_image(rgb_img)
    entropy = _entropy(gray)
    grad = _gradient_density(gray)
    edge = _edge_density(gray)
    largest_tex, tile_cov = _largest_texture_region_ratio(gray)
    background = float(np.clip(1.0 - tile_cov, 0.0, 1.0))
    ui = _has_ui_chrome(rgb)
    band_delta = _band_color_delta(rgb)
    looks_scene = ImagePreprocessor._looks_like_scene_photo(rgb_img)

    is_catalog = _is_catalog_sheet(
        aspect=aspect,
        width=width,
        height=height,
        white_border=catalog.white_border_ratio,
        text_score=catalog.text_region_score,
        band_delta=band_delta,
        has_preview_grid=bool(catalog.has_preview_grid),
    )
    is_phone = ui and (aspect < 0.85 or height >= 1000)

    # Room / installation: scene-like framing + strong ceiling/floor color
    # shift. Do NOT use aspect alone — tall catalogue panel crops are also
    # non-square but have flat band_delta (~1) vs rooms (~90+).
    is_room = (
        not is_catalog
        and not is_phone
        and looks_scene
        and band_delta >= 35.0
    )

    is_clean = (
        not is_catalog
        and not is_phone
        and not is_room
        and 0.85 <= aspect <= 1.18
        and catalog.texture_richness >= 0.12
        and band_delta < 25.0
    )

    is_partial = (
        not is_catalog
        and not is_phone
        and not is_room
        and not is_clean
        and catalog.texture_richness >= 0.12
    )

    if is_catalog:
        kind, confidence = QueryKind.CATALOG_SHEET, 0.88
    elif is_phone:
        kind, confidence = QueryKind.PHONE_SCREENSHOT, 0.85
    elif is_room:
        kind, confidence = QueryKind.ROOM_SCENE, 0.85
    elif is_clean:
        kind, confidence = QueryKind.CLEAN_TILE, 0.80
    elif is_partial:
        kind, confidence = QueryKind.PARTIAL_CROP, 0.65
    else:
        kind, confidence = QueryKind.UNKNOWN, 0.40

    return QueryAnalysis(
        kind=kind,
        width=width,
        height=height,
        aspect_ratio=float(aspect),
        tile_coverage_ratio=float(tile_cov),
        background_ratio=background,
        entropy=entropy,
        gradient_density=grad,
        texture_density=float(catalog.texture_richness),
        edge_density=edge,
        white_border_ratio=float(catalog.white_border_ratio),
        largest_texture_area_ratio=float(largest_tex),
        has_ui_chrome=ui,
        text_region_score=float(catalog.text_region_score),
        has_preview_grid=bool(catalog.has_preview_grid),
        band_color_delta=float(band_delta),
        confidence=float(confidence),
    )
