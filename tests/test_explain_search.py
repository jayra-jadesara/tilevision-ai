"""Tests for scripts/explain_search.py."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.explain_search import explain_search, resolve_catalog_paths
from src.ai.candidate_filter import CandidateFilter
from src.ai.models import TileFeatures
from src.ai.pattern_classifier import PatternClassifier
from src.ai.reranker import HybridReRanker
from src.core.models import TileImage
from src.core.use_cases.search_tiles import SearchTilesUseCase


def _features(emb: list[float]) -> TileFeatures:
    vec = np.asarray(emb, dtype=np.float32)
    vec /= float(np.linalg.norm(vec))
    return TileFeatures(
        embedding=vec,
        color_histogram=np.full(8, 1.0 / 8, dtype=np.float32),
        texture_histogram=np.full(8, 1.0 / 8, dtype=np.float32),
        edge_histogram=np.full(8, 1.0 / 8, dtype=np.float32),
        pattern_features=np.zeros(12, dtype=np.float32),
        dominant_color=(200, 200, 200),
        width=64,
        height=64,
    )


def test_resolve_catalog_paths_standard_layout(tmp_path: Path):
    (tmp_path / "database").mkdir()
    (tmp_path / "index").mkdir()
    (tmp_path / "thumbnails").mkdir()
    (tmp_path / "database" / "tiles.db").write_text("db")
    (tmp_path / "index" / "tiles.index").write_text("idx")

    db, idx, thumbs = resolve_catalog_paths(tmp_path)
    assert db.endswith("tiles.db")
    assert idx.endswith("tiles.index")
    assert thumbs.endswith("thumbnails")


def test_explain_search_reports_component_scores(tmp_path: Path):
    query = tmp_path / "query.jpg"
    Image.new("RGB", (400, 380), color=(180, 170, 160)).save(query)

    tile_a = TileImage(
        id=1,
        file_path=str(tmp_path / "a.jpg"),
        file_name="a.jpg",
        file_size=1,
        dimensions="64x64",
        features=_features([1.0, 0.0, 0.0, 0.0]),
    )
    tile_b = TileImage(
        id=2,
        file_path=str(tmp_path / "b.jpg"),
        file_name="b.jpg",
        file_size=1,
        dimensions="64x64",
        features=_features([0.7, 0.7, 0.0, 0.0]),
    )

    query_features = _features([1.0, 0.0, 0.0, 0.0])
    emb_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    emb_b = np.array([0.8, 0.2, 0.0, 0.0], dtype=np.float32)

    repo = MagicMock()
    repo.get_by_ids.return_value = [tile_a, tile_b]

    index = MagicMock()
    index.get_total_count.return_value = 2

    extractor = MagicMock()
    extractor.extract_for_search.return_value = (
        query_features,
        [emb_a, emb_b],
    )

    use_case = SearchTilesUseCase.__new__(SearchTilesUseCase)
    use_case._repo = repo
    use_case._index = index
    use_case._feature_extractor = extractor
    use_case._reranker = HybridReRanker()
    use_case._enable_orb_verification = False
    use_case._orb_verifier = None
    use_case._search_faiss_multi_crop = MagicMock(
        return_value=(
            [1, 2],
            {1: 0.95, 2: 0.62},
            {1: 1, 2: 0},
        )
    )

    report = explain_search(use_case, query, top_k=2)

    assert report.query_kind
    assert report.query_view_count == 2
    assert len(report.candidates) >= 2
    assert report.candidates[0].tile_id == 1
    assert report.candidates[0].winning_view_index == 1

    reranker = HybridReRanker()
    qpt = PatternClassifier.classify(query_features)
    cpt = PatternClassifier.classify(tile_a.features)
    expected = reranker.score(
        query_features,
        tile_a.features,
        query_pattern_type=qpt,
        candidate_pattern_type=cpt,
    )
    assert report.candidates[0].embedding == pytest.approx(expected.embedding)
    assert report.candidates[0].color == pytest.approx(expected.color)
    assert report.candidates[0].texture == pytest.approx(expected.texture)
    assert report.candidates[0].edge == pytest.approx(expected.edge)
    assert report.candidates[0].pattern == pytest.approx(expected.pattern)
    assert report.candidates[0].color_penalty == pytest.approx(
        CandidateFilter.dominant_color_penalty(query_features, tile_a.features)
    )
