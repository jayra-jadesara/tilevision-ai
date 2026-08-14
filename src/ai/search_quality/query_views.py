"""
Adaptive query-view generation (search-only, no index changes).

Selects preprocessing / crop candidates from QueryAnalysis. All crops are
OpenCV/PIL heuristics — no extra AI models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.ai.preprocess.fast_tile_crop import (
    isolate_tile_region,
    list_tile_region_candidates,
)
from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.search_quality.query_analyzer import QueryAnalysis, QueryKind, analyze_query

logger = logging.getLogger("tilevision.ai.query_views")


@dataclass(frozen=True, slots=True)
class QueryViewPlan:
    kind: QueryKind
    max_views: int
    strip_ui: bool
    isolate_scene: bool
    prefer_panel: bool
    preserve_aspect: bool


def plan_query_views(analysis: QueryAnalysis, *, max_views_cap: int = 3) -> QueryViewPlan:
    """Map analysis → preprocessing plan. Caps keep Mac/Windows CPU ready."""
    cap = max(1, int(max_views_cap))
    if analysis.kind == QueryKind.CLEAN_TILE:
        return QueryViewPlan(
            kind=analysis.kind,
            max_views=1,
            strip_ui=False,
            isolate_scene=False,
            prefer_panel=False,
            preserve_aspect=True,
        )
    if analysis.kind == QueryKind.CATALOG_SHEET:
        return QueryViewPlan(
            kind=analysis.kind,
            max_views=min(2, cap),
            strip_ui=False,
            isolate_scene=False,
            prefer_panel=True,
            preserve_aspect=True,
        )
    if analysis.kind == QueryKind.PHONE_SCREENSHOT:
        return QueryViewPlan(
            kind=analysis.kind,
            max_views=min(2, cap),
            strip_ui=True,
            isolate_scene=True,
            prefer_panel=False,
            preserve_aspect=True,
        )
    if analysis.kind == QueryKind.ROOM_SCENE:
        return QueryViewPlan(
            kind=analysis.kind,
            max_views=min(3, cap),
            strip_ui=False,
            isolate_scene=True,
            prefer_panel=False,
            preserve_aspect=False,
        )
    if analysis.kind == QueryKind.PARTIAL_CROP:
        return QueryViewPlan(
            kind=analysis.kind,
            max_views=min(2, cap),
            strip_ui=False,
            isolate_scene=False,
            prefer_panel=False,
            preserve_aspect=True,
        )
    # Unknown: conservative single view, isolate only if background-heavy.
    return QueryViewPlan(
        kind=analysis.kind,
        max_views=1 if analysis.background_ratio < 0.50 else min(2, cap),
        strip_ui=False,
        isolate_scene=analysis.background_ratio >= 0.50,
        prefer_panel=False,
        preserve_aspect=True,
    )


def strip_phone_ui(image: Image.Image) -> Image.Image:
    """Remove dark status-bar / chrome bands common in phone screenshots."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    arr = np.asarray(rgb)
    top_h = max(8, h // 14)
    bot_h = max(8, h // 18)
    top_mean = float(arr[:top_h].mean())
    bot_mean = float(arr[h - bot_h :].mean())
    top = top_h if top_mean < 50.0 else 0
    bot = bot_h if bot_mean < 50.0 else 0
    # Side letterbox (common in screenshots)
    left = 0
    right = w
    col_mean = arr.mean(axis=(0, 2)) if arr.ndim == 3 else arr.mean(axis=0)
    # Use vertical mean of left/right strips
    left_strip = float(arr[:, : max(4, w // 30)].mean())
    right_strip = float(arr[:, w - max(4, w // 30) :].mean())
    if left_strip < 35.0:
        left = max(4, w // 28)
    if right_strip < 35.0:
        right = w - max(4, w // 28)
    if top == 0 and bot == 0 and left == 0 and right == w:
        return rgb
    cropped = rgb.crop((left, top, right, h - bot if bot else h))
    if min(cropped.size) < 64:
        return rgb
    return cropped


def _center_crop(image: Image.Image, ratio: float) -> Image.Image:
    w, h = image.size
    cw, ch = max(1, int(w * ratio)), max(1, int(h * ratio))
    left, top = (w - cw) // 2, (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch))


def _texture_window(image: Image.Image) -> Image.Image | None:
    """Largest high-gradient window via integral of Sobel magnitude."""
    import cv2

    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    if min(h, w) < 80:
        return None
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    win = max(64, int(min(h, w) * 0.45))
    integral = cv2.integral(mag)
    best = -1.0
    best_box = None
    step = max(8, win // 6)
    for y in range(0, h - win + 1, step):
        for x in range(0, w - win + 1, step):
            x2, y2 = x + win, y + win
            s = (
                integral[y2, x2]
                - integral[y, x2]
                - integral[y2, x]
                + integral[y, x]
            )
            if s > best:
                best = float(s)
                best_box = (x, y, x2, y2)
    if best_box is None:
        return None
    return image.crop(best_box)


def collect_query_crop_pils(
    image: Image.Image,
    *,
    analysis: QueryAnalysis | None = None,
    max_views_cap: int = 3,
) -> tuple[QueryAnalysis, list[Image.Image]]:
    """
    Build ordered PIL crops for query embedding (best-first).

    Index is never modified. Caller embeds each crop and FAISS-merges by MAX.
    """
    rgb = ImagePreprocessor.to_rgb(image)
    rgb = ImagePreprocessor.trim_uniform_borders(rgb)
    analysis = analysis or analyze_query(rgb)
    plan = plan_query_views(analysis, max_views_cap=max_views_cap)

    working = rgb
    if plan.strip_ui:
        working = strip_phone_ui(working)
        working = ImagePreprocessor.trim_uniform_borders(working)

    crops: list[Image.Image] = []

    if plan.prefer_panel:
        panel = ImagePreprocessor.primary_texture_panel(working)
        if panel is not None:
            crops.append(panel)
            if plan.max_views >= 2 and min(panel.size) >= 200:
                crops.append(_center_crop(panel, 0.72))
        # Always keep a full-sheet view as fallback for layout queries.
        crops.append(working)
    elif plan.kind == QueryKind.PARTIAL_CROP:
        content = ImagePreprocessor.crop_to_content_region(
            working,
            min_margin_ratio=0.02,
        )
        crops.append(content)
        if plan.max_views >= 2:
            crops.append(_center_crop(content, 0.82))
    elif plan.isolate_scene:
        primary = isolate_tile_region(working)
        crops.append(primary.image)
        if plan.max_views >= 2:
            for cand in list_tile_region_candidates(working, limit=plan.max_views + 1):
                if cand.box == primary.box:
                    continue
                crops.append(cand.image)
                if len(crops) >= plan.max_views:
                    break
        if plan.max_views >= 3 and len(crops) < plan.max_views:
            tex = _texture_window(working)
            if tex is not None:
                crops.append(tex)
        if len(crops) < plan.max_views:
            crops.append(_center_crop(working, 0.55))
    else:
        # Clean / unknown. Rotation-expand and perspective fills create large
        # white corners (white_border_ratio ≫ 0) — aggressive content crop
        # alone regressed Recall@5 by ~30pp on ±10° rotations. Prefer
        # OpenCV isolation of the face inside the white frame (same idea as
        # the v1.2.31 looks_like_scene path), fall back to content crop.
        from src.ai.search_quality.query_analyzer import QueryKind as _QK

        high_frame = analysis.white_border_ratio >= 0.25
        if high_frame or (
            analysis.kind == _QK.UNKNOWN
            and ImagePreprocessor._looks_like_scene_photo(working)
        ):
            iso = isolate_tile_region(working)
            crops.append(iso.image)
            if plan.max_views >= 2:
                content = ImagePreprocessor.crop_to_content_region(
                    working, min_margin_ratio=0.05
                )
                crops.append(content)
        else:
            content = ImagePreprocessor.crop_to_content_region(
                working, min_margin_ratio=0.05
            )
            crops.append(content)

    # Deduplicate near-identical sizes / boxes
    unique: list[Image.Image] = []
    seen: list[tuple[int, int]] = []
    for crop in crops:
        key = crop.size
        if key in seen and len(unique) > 0:
            # Allow one duplicate size only if pixels differ enough.
            prev = np.asarray(unique[-1].resize(key), dtype=np.int16)
            cur = np.asarray(crop.resize(key), dtype=np.int16)
            if float(np.mean(np.abs(prev - cur))) < 4.0:
                continue
        seen.append(key)
        unique.append(crop)
        if len(unique) >= plan.max_views:
            break

    if not unique:
        unique = [working]

    logger.info(
        "Query view plan: kind=%s views=%d isolate=%s panel=%s ui=%s",
        analysis.kind.value,
        len(unique),
        plan.isolate_scene,
        plan.prefer_panel,
        plan.strip_ui,
    )
    return analysis, unique


def collect_crop_tool_pils(
    image: Image.Image,
    *,
    max_views_cap: int = 2,
) -> list[Image.Image]:
    """
    Views for Auto / Precise / Manual crop-tool outputs.

    The file is already an isolated tile. Do not re-run scene isolation or
    aggressive content-crop (that shrinks an already-tight crop further).
    Primary = light border trim; optional second view = 82% center so FAISS
    can match a tighter index panel.
    """
    working = ImagePreprocessor.to_rgb(image)
    working = ImagePreprocessor.trim_uniform_borders(working)
    crops: list[Image.Image] = [working]
    cap = max(1, int(max_views_cap))
    if cap >= 2 and min(working.size) >= 200:
        crops.append(_center_crop(working, 0.82))
    logger.info(
        "Crop-tool query views: n=%d sizes=%s",
        len(crops),
        [c.size for c in crops],
    )
    return crops
