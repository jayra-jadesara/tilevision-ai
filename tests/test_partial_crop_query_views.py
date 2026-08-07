"""Tests for PARTIAL_CROP multi-view query planning."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.search_quality.query_analyzer import QueryAnalysis, QueryKind
from src.ai.search_quality.query_views import collect_query_crop_pils, plan_query_views
from src.core.use_cases.search_tiles import SearchTilesUseCase


def _partial_crop_image(seed: int = 4) -> Image.Image:
    rng = np.random.default_rng(seed)
    tile = rng.integers(120, 220, (520, 480, 3), dtype=np.uint8)
    canvas = np.full((600, 620, 3), 235, dtype=np.uint8)
    canvas[40:560, 70:550] = tile
    return Image.fromarray(canvas)


def test_partial_crop_plan_requests_up_to_three_views():
    analysis = QueryAnalysis(
        kind=QueryKind.PARTIAL_CROP,
        width=620,
        height=600,
        aspect_ratio=620 / 600,
        tile_coverage_ratio=0.7,
        background_ratio=0.3,
        entropy=0.5,
        gradient_density=0.4,
        texture_density=0.3,
        edge_density=0.2,
        white_border_ratio=0.08,
        largest_texture_area_ratio=0.6,
        has_ui_chrome=False,
        text_region_score=0.02,
        has_preview_grid=False,
        band_color_delta=12.0,
        confidence=0.65,
    )
    plan = plan_query_views(analysis, max_views_cap=3)
    assert plan.kind == QueryKind.PARTIAL_CROP
    assert plan.max_views == 3
    assert plan.isolate_scene is False


def test_partial_crop_collects_complementary_views():
    img = _partial_crop_image()
    analysis = QueryAnalysis(
        kind=QueryKind.PARTIAL_CROP,
        width=img.size[0],
        height=img.size[1],
        aspect_ratio=img.size[0] / img.size[1],
        tile_coverage_ratio=0.7,
        background_ratio=0.3,
        entropy=0.5,
        gradient_density=0.4,
        texture_density=0.3,
        edge_density=0.2,
        white_border_ratio=0.08,
        largest_texture_area_ratio=0.6,
        has_ui_chrome=False,
        text_region_score=0.02,
        has_preview_grid=False,
        band_color_delta=12.0,
        confidence=0.65,
    )
    _, crops = collect_query_crop_pils(img, analysis=analysis, max_views_cap=3)
    assert len(crops) == 3
    sizes = {c.size for c in crops}
    assert len(sizes) >= 2


def test_partial_crop_feature_extractor_uses_multi_view_path():
    from src.ai.feature_extractor import ExtractTimings, FeatureExtractor

    img = _partial_crop_image()
    analysis = QueryAnalysis(
        kind=QueryKind.PARTIAL_CROP,
        width=img.size[0],
        height=img.size[1],
        aspect_ratio=img.size[0] / img.size[1],
        tile_coverage_ratio=0.7,
        background_ratio=0.3,
        entropy=0.5,
        gradient_density=0.4,
        texture_density=0.3,
        edge_density=0.2,
        white_border_ratio=0.08,
        largest_texture_area_ratio=0.6,
        has_ui_chrome=False,
        text_region_score=0.02,
        has_preview_grid=False,
        band_color_delta=12.0,
        confidence=0.65,
    )

    class _FakeEmbedder:
        def extract_from_preprocessed(self, _view, *, for_query=True):
            return np.array([1.0, 0.0], dtype=np.float32)

    fx = FeatureExtractor.__new__(FeatureExtractor)
    fx._embedder = _FakeEmbedder()
    fx._last_timings = ExtractTimings(0.0, 0.0, 0.0, 0.0)

    with patch(
        "src.ai.search_quality.query_analyzer.analyze_query",
        return_value=analysis,
    ):
        with patch.object(
            fx,
            "_fuse_query_embeddings",
            side_effect=lambda primary, embs, elapsed: type(
                "F",
                (),
                {"embedding": embs[0]},
            )(),
        ):
            with patch(
                "src.ai.preprocess.image_preprocessor.ImagePreprocessor._finalize_query_pil",
                side_effect=lambda crop, **kwargs: type(
                    "P",
                    (),
                    {"pil": crop},
                )(),
            ):
                _feats, embeddings = fx.extract_for_search("query.jpg", preloaded=img)

    assert len(embeddings) > 1


def test_faiss_max_merge_across_partial_crop_views():
    index_calls = []

    class _FakeIndex:
        def search_vectors(self, _vector, _top_k):
            index_calls.append(tuple(_vector))
            if len(index_calls) == 1:
                return [10, 20], [0.55, 0.70]
            return [20, 30], [0.82, 0.60]

    use_case = SearchTilesUseCase.__new__(SearchTilesUseCase)
    use_case._index = _FakeIndex()
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    ordered, scores, view_map = use_case._search_faiss_multi_crop(embeddings, 10)

    assert ordered[0] == 20
    assert scores[20] == pytest.approx(0.82)
    assert view_map[20] == 1
    assert len(index_calls) == 2
