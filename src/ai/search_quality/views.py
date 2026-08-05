"""
Index-time multi-view builders for search-quality bakeoffs.

Strategies are isolated so each can be benchmarked independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from PIL import Image

from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.search_quality.image_analysis import ImageAnalysis, analyze_image


class IndexViewType(str, Enum):
    PRIMARY = "primary"
    CENTER = "center"
    ADAPTIVE = "adaptive"
    TEXTURE = "texture"
    PANEL = "panel"
    PANEL_CENTER = "panel_center"


@dataclass(frozen=True, slots=True)
class IndexView:
    view_type: IndexViewType
    image: Image.Image
    crop_box: tuple[int, int, int, int]  # left, top, right, bottom in source
    quality_score: float
    confidence: float


class IndexStrategy(str, Enum):
    """Bakeoff strategies from the search-accuracy project brief."""

    A_PRIMARY_ONLY = "A_primary_only"
    B_FULL_CENTER = "B_full_center"
    C_FULL_ADAPTIVE = "C_full_adaptive"
    D_FULL_TEXTURE = "D_full_texture"
    E_HEURISTIC_MULTIVIEW = "E_heuristic_multiview"
    # Current shipped production path (v1.2.29) for regression comparison.
    PRODUCTION_V8 = "production_v8"


def _full_box(image: Image.Image) -> tuple[int, int, int, int]:
    w, h = image.size
    return (0, 0, w, h)


def _center_box(image: Image.Image, ratio: float) -> tuple[int, int, int, int]:
    w, h = image.size
    cw, ch = max(1, int(w * ratio)), max(1, int(h * ratio))
    left, top = (w - cw) // 2, (h - ch) // 2
    return (left, top, left + cw, top + ch)


def _texture_rich_crop(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """
    Pick the highest local-variance window (approx 50% of min side).

    Lightweight sliding-window on a downscaled gray map — no blind quadrants.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    side = max(64, int(min(h, w) * 0.50))
    if side >= min(h, w):
        box = _full_box(image)
        return image, box

    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    scale = 256.0 / max(h, w)
    small = cv2.resize(
        gray,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    sh, sw = small.shape
    win = max(8, int(side * scale))
    best_val = -1.0
    best_yx = (0, 0)
    step = max(1, win // 3)
    for y in range(0, max(1, sh - win + 1), step):
        for x in range(0, max(1, sw - win + 1), step):
            patch = small[y : y + win, x : x + win]
            val = float(patch.var())
            if val > best_val:
                best_val = val
                best_yx = (y, x)
    y, x = best_yx
    left = int(x / scale)
    top = int(y / scale)
    left = min(max(0, left), w - side)
    top = min(max(0, top), h - side)
    box = (left, top, left + side, top + side)
    return image.crop(box), box


def _adaptive_content_crop(
    image: Image.Image,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Content-region crop via existing preprocessor heuristics."""
    trimmed = ImagePreprocessor.trim_uniform_borders(image)
    cropped = ImagePreprocessor.crop_to_content_region(trimmed, min_margin_ratio=0.05)
    # Map back approximately: if sizes match original, full box.
    if cropped.size == image.size:
        return image, _full_box(image)
    # Content crop loses absolute offset after trim; use center of trimmed area.
    tw, th = trimmed.size
    cw, ch = cropped.size
    # Approximate: content centered in trimmed frame after border trim.
    # Use focus center of original at cropped/original ratio as stable proxy.
    ratio = min(cw / max(tw, 1), ch / max(th, 1))
    ratio = float(np.clip(ratio, 0.45, 0.95))
    box = _center_box(image, ratio)
    return image.crop(box), box


def build_index_views(
    image: Image.Image,
    strategy: IndexStrategy | str,
    *,
    analysis: ImageAnalysis | None = None,
) -> list[IndexView]:
    """
    Build index views for a strategy. Primary is always first.

    Strategies B–D always add their named aux when the crop differs enough
    in area from the full frame. Strategy E only adds views the heuristics
    mark as beneficial.
    """
    if not isinstance(strategy, IndexStrategy):
        strategy = IndexStrategy(strategy)

    image = image.convert("RGB")
    analysis = analysis or analyze_image(image)
    views: list[IndexView] = [
        IndexView(
            view_type=IndexViewType.PRIMARY,
            image=image,
            crop_box=_full_box(image),
            quality_score=analysis.quality_score,
            confidence=1.0,
        )
    ]

    def add(
        view_type: IndexViewType,
        crop: Image.Image,
        box: tuple[int, int, int, int],
        confidence: float,
    ) -> None:
        area = (box[2] - box[0]) * (box[3] - box[1])
        full = image.size[0] * image.size[1]
        if area >= full * 0.92:
            return
        if min(crop.size) < 64:
            return
        views.append(
            IndexView(
                view_type=view_type,
                image=crop,
                crop_box=box,
                quality_score=analysis.quality_score * confidence,
                confidence=confidence,
            )
        )

    if strategy == IndexStrategy.A_PRIMARY_ONLY:
        return views

    if strategy == IndexStrategy.B_FULL_CENTER:
        box = _center_box(image, 0.50)
        add(IndexViewType.CENTER, image.crop(box), box, 0.85)
        return views

    if strategy == IndexStrategy.C_FULL_ADAPTIVE:
        crop, box = _adaptive_content_crop(image)
        add(IndexViewType.ADAPTIVE, crop, box, 0.80)
        return views

    if strategy == IndexStrategy.D_FULL_TEXTURE:
        crop, box = _texture_rich_crop(image)
        add(IndexViewType.TEXTURE, crop, box, 0.80)
        return views

    if strategy == IndexStrategy.E_HEURISTIC_MULTIVIEW:
        if analysis.left_panel_beneficial:
            panel = ImagePreprocessor.primary_texture_panel(image)
            if panel is not None:
                # Panel is a left crop; approximate box as left 45%.
                split = max(1, int(image.size[0] * 0.45))
                box = (0, 0, split, image.size[1])
                add(IndexViewType.PANEL, panel, box, 0.95)
                if min(panel.size) >= 200:
                    pbox = _center_box(panel, 0.72)
                    # Nest panel-center coords into source approx.
                    src_box = (
                        box[0] + pbox[0],
                        box[1] + pbox[1],
                        box[0] + pbox[2],
                        box[1] + pbox[3],
                    )
                    add(
                        IndexViewType.PANEL_CENTER,
                        panel.crop(pbox),
                        src_box,
                        0.90,
                    )
        if analysis.center_crop_beneficial:
            for ratio, conf in ((0.50, 0.85), (0.40, 0.80)):
                box = _center_box(image, ratio)
                before = len(views)
                add(IndexViewType.CENTER, image.crop(box), box, conf)
                if len(views) > before:
                    break
        elif analysis.kind.value == "bordered_tile":
            crop, box = _adaptive_content_crop(image)
            add(IndexViewType.ADAPTIVE, crop, box, 0.75)
            # If borders were trimmed, still add a center texture view when
            # the face is large and textured (customer 600×600 crops).
            if analysis.texture_richness >= 0.15 and min(image.size) >= 400:
                for ratio, conf in ((0.50, 0.80), (0.40, 0.75)):
                    cbox = _center_box(image, ratio)
                    before = len(views)
                    add(IndexViewType.CENTER, image.crop(cbox), cbox, conf)
                    if len(views) > before:
                        break
        return views

    if strategy == IndexStrategy.PRODUCTION_V8:
        # Mirror feature_extractor.extract_index_vectors view selection.
        panel = ImagePreprocessor.primary_texture_panel(image)
        if panel is not None:
            split = max(1, int(image.size[0] * 0.45))
            box = (0, 0, split, image.size[1])
            add(IndexViewType.PANEL, panel, box, 0.95)
            if min(panel.size) >= 200:
                pbox = _center_box(panel, 0.72)
                src_box = (
                    box[0] + pbox[0],
                    box[1] + pbox[1],
                    box[0] + pbox[2],
                    box[1] + pbox[3],
                )
                add(IndexViewType.PANEL_CENTER, panel.crop(pbox), src_box, 0.90)
        elif min(image.size) >= 400:
            for ratio in (0.50, 0.40):
                box = _center_box(image, ratio)
                before = len(views)
                add(IndexViewType.CENTER, image.crop(box), box, 0.85)
                if len(views) > before:
                    break
        return views

    raise ValueError(f"Unknown strategy: {strategy}")


STRATEGY_BUILDERS: dict[IndexStrategy, Callable[..., list[IndexView]]] = {
    s: (lambda img, s=s: build_index_views(img, s)) for s in IndexStrategy
}
