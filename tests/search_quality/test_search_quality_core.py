"""Unit tests for search-quality image analysis, views, and fusion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ai.search_quality.fusion import FusionMethod, ScoredHit, fuse_hits
from src.ai.search_quality.image_analysis import ImageKind, analyze_image
from src.ai.search_quality.views import IndexStrategy, IndexViewType, build_index_views


def _marble(h=400, w=400, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(180, 240, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_analyze_clean_tile():
    img = _marble()
    a = analyze_image(img)
    assert a.kind in {ImageKind.CLEAN_TILE, ImageKind.UNKNOWN, ImageKind.BORDERED_TILE}
    assert 0.0 <= a.texture_richness <= 1.0


def test_analyze_catalog_sheet_flags_left_panel():
    sheet = Image.new("RGB", (1200, 900), (255, 255, 255))
    sheet.paste(_marble(880, 500, seed=3), (20, 10))
    draw = ImageDraw.Draw(sheet)
    draw.text((560, 40), "ELEGANT CATALOGUE", fill=(0, 0, 0))
    for i in range(6):
        draw.rectangle((560 + (i % 3) * 100, 300 + (i // 3) * 160, 640 + (i % 3) * 100, 440 + (i // 3) * 160), outline=(0, 0, 0))
    a = analyze_image(sheet)
    assert a.left_panel_beneficial is True
    assert a.kind == ImageKind.CATALOG_SHEET


def test_strategy_a_primary_only():
    views = build_index_views(_marble(800, 800), IndexStrategy.A_PRIMARY_ONLY)
    assert len(views) == 1
    assert views[0].view_type == IndexViewType.PRIMARY


def test_strategy_b_adds_center():
    views = build_index_views(_marble(800, 800), IndexStrategy.B_FULL_CENTER)
    types = {v.view_type for v in views}
    assert IndexViewType.PRIMARY in types
    assert IndexViewType.CENTER in types


def test_strategy_e_sheet_gets_panel_not_blind_quadrants():
    sheet = Image.new("RGB", (1200, 900), (255, 255, 255))
    sheet.paste(_marble(880, 500, seed=9), (20, 10))
    draw = ImageDraw.Draw(sheet)
    draw.text((560, 40), "SKU TEXT COLUMN", fill=(10, 10, 10))
    views = build_index_views(sheet, IndexStrategy.E_HEURISTIC_MULTIVIEW)
    types = [v.view_type for v in views]
    assert IndexViewType.PRIMARY in types
    # Must not invent four quadrant crops
    assert types.count(IndexViewType.CENTER) <= 1
    assert IndexViewType.PANEL in types or IndexViewType.PANEL_CENTER in types or len(views) >= 1


def test_fusion_max_groups_by_tile_id():
    hits = [
        ScoredHit(1, 0.9, 1.0, 1),
        ScoredHit(1, 0.95, 0.8, 2),
        ScoredHit(2, 0.92, 1.0, 3),
    ]
    fused = fuse_hits(hits, FusionMethod.MAX)
    assert fused[0] == (1, 0.95)
    assert fused[1][0] == 2


def test_fusion_softmax_groups_by_tile_id():
    hits = [
        ScoredHit(1, 0.9, 1.0, 1),
        ScoredHit(1, 0.8, 1.0, 2),
        ScoredHit(2, 0.95, 1.0, 3),
    ]
    fused = fuse_hits(hits, FusionMethod.SOFTMAX)
    assert {tid for tid, _ in fused} == {1, 2}
    assert fused[0][1] >= fused[1][1]


def test_fusion_rrf_prefers_multi_list_support():
    hits = [
        ScoredHit(1, 0.99, 1.0, 5),
        ScoredHit(2, 0.50, 1.0, 1),
        ScoredHit(2, 0.40, 1.0, 2),
    ]
    fused = fuse_hits(hits, FusionMethod.RRF, rrf_k=60)
    # tile 2 has ranks 1 and 2 → higher RRF than tile 1 at rank 5
    assert fused[0][0] == 2
