"""
Bakeoff ``production_v8`` must stay locked to real index-time view selection.

Regression for the silent-drift class: ``production_v8`` in views.py used to
maintain a parallel fork of ``extract_index_vectors`` / ``prepare_index_primary``
(missing force-adaptive, ungated ``primary_texture_panel``). Same spirit as
``test_batch_index_parity.py`` after the batch-vs-standalone fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.preprocess.index_primary import prepare_index_primary
from src.ai.search_quality.image_analysis import analyze_image
from src.ai.search_quality.views import IndexStrategy, IndexViewType, build_index_views
from tests.test_crop_search_consistency import _make_catalog_sheet, _make_marble


def _view_signature(views) -> list[tuple[str, tuple[int, int], tuple[int, int, int, int]]]:
    return [
        (v.view_type.value, v.image.size, v.crop_box)
        for v in views
    ]


def test_production_v8_views_match_prepare_index_primary_on_catalog_sheet(tmp_path):
    """Catalog sheet: production_v8 must include panel + adaptive like production."""
    sheet_path, _ = _make_catalog_sheet(tmp_path)
    raw = ImagePreprocessor.to_rgb(ImagePreprocessor.load(sheet_path))
    analysis = analyze_image(raw)
    assert analysis.left_panel_beneficial is True

    v8 = build_index_views(raw, IndexStrategy.PRODUCTION_V8, analysis=analysis)
    prep = prepare_index_primary(sheet_path)

    assert _view_signature(v8) == _view_signature(prep.views)
    assert IndexViewType.PANEL in {v.view_type for v in v8}
    assert IndexViewType.ADAPTIVE in {v.view_type for v in v8}
    assert prep.primary_source == "panel"
    assert prep.panel is not None

    v8_panel = next(v for v in v8 if v.view_type == IndexViewType.PANEL)
    assert np.array_equal(np.asarray(v8_panel.image), np.asarray(prep.panel))


def test_production_v8_matches_strategy_e_view_plan(tmp_path):
    """PRODUCTION_V8 is Strategy E — not a separate v1.2.29 fork."""
    sheet_path, _ = _make_catalog_sheet(tmp_path)
    raw = ImagePreprocessor.to_rgb(ImagePreprocessor.load(sheet_path))
    analysis = analyze_image(raw)

    v8 = build_index_views(raw, IndexStrategy.PRODUCTION_V8, analysis=analysis)
    e = build_index_views(
        raw, IndexStrategy.E_HEURISTIC_MULTIVIEW, analysis=analysis
    )
    assert _view_signature(v8) == _view_signature(e)


def test_production_v8_clean_tile_matches_prepare_index_primary(tmp_path):
    tile = _make_marble(600, 600, seed=3)
    path = tmp_path / "clean.jpg"
    tile.save(path)
    raw = ImagePreprocessor.to_rgb(ImagePreprocessor.load(path))
    analysis = analyze_image(raw)

    v8 = build_index_views(raw, IndexStrategy.PRODUCTION_V8, analysis=analysis)
    prep = prepare_index_primary(path)
    assert _view_signature(v8) == _view_signature(prep.views)
    assert prep.primary_source == "full_sheet"


def test_production_v8_gates_panel_on_left_panel_beneficial():
    """
    Wide image without marketing column must not invent a panel view.

    The old production_v8 fork called ``primary_texture_panel`` ungated;
    production only isolates when ``left_panel_beneficial``.
    """
    # Square-ish textured face — not a marketing sheet.
    img = _make_marble(800, 800, seed=5)
    analysis = analyze_image(img)
    assert analysis.left_panel_beneficial is False

    v8 = build_index_views(img, IndexStrategy.PRODUCTION_V8, analysis=analysis)
    types = {v.view_type for v in v8}
    assert IndexViewType.PANEL not in types
    assert IndexViewType.PANEL_CENTER not in types


def test_query_side_partial_crop_out_of_scope_for_production_v8():
    """
    PARTIAL_CROP / already_clean / crop-tool batching are query-side only.

    production_v8 is an index strategy — those PR #46/#47 paths live in
    extract_for_search / query preprocess, not in build_index_views.
    """
    views_src = Path(
        Path(__file__).resolve().parents[2]
        / "src"
        / "ai"
        / "search_quality"
        / "views.py"
    ).read_text(encoding="utf-8")
    index_primary_src = Path(
        Path(__file__).resolve().parents[2]
        / "src"
        / "ai"
        / "preprocess"
        / "index_primary.py"
    ).read_text(encoding="utf-8")
    for src in (views_src, index_primary_src):
        assert "already_clean" not in src
        assert "PARTIAL_CROP" not in src
        assert "partial_crop" not in src
