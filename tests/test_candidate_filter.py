"""Tests for soft candidate color compatibility."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.candidate_filter import CandidateFilter
from src.ai.models import TileFeatures
from src.ai.reranker import HybridReRanker


def _features(dominant_color: tuple[int, int, int]) -> TileFeatures:
    return TileFeatures(
        embedding=np.ones(4, dtype=np.float32),
        color_histogram=np.ones(16, dtype=np.float32),
        texture_histogram=np.ones(16, dtype=np.float32),
        edge_histogram=np.ones(36, dtype=np.float32),
        pattern_features=np.zeros(12, dtype=np.float32),
        dominant_color=dominant_color,
        width=32,
        height=32,
    )


def test_similar_colors_have_zero_penalty():
    query = _features((200, 200, 200))
    candidate = _features((205, 203, 198))
    assert CandidateFilter.dominant_color_penalty(query, candidate) == 0.0


def test_very_different_colors_get_max_penalty():
    query = _features((255, 255, 255))
    candidate = _features((0, 0, 0))
    penalty = CandidateFilter.dominant_color_penalty(query, candidate)
    assert penalty == -CandidateFilter.COLOR_PENALTY_MAX


def test_filter_does_not_remove_candidates():
    from src.core.models import TileImage

    query = _features((255, 255, 255))
    tile = TileImage(
        id=1,
        file_path="x.jpg",
        file_name="x.jpg",
        file_size=1024,
        dimensions="32x32",
        brand="A",
        category="Floor",
        color="White",
        size="60x60",
        product_code="X",
        is_indexed=True,
        features=_features((0, 0, 0)),
    )
    kept = CandidateFilter.filter(query, [tile])
    assert len(kept) == 1


def test_color_penalty_reduces_reranker_final_score():
    reranker = HybridReRanker()
    query = _features((255, 255, 255))
    same_color = _features((250, 250, 250))
    opposite = _features((0, 0, 0))

    high_score = reranker.score(query, same_color).final
    low_score = reranker.score(query, opposite).final
    assert low_score < high_score
    assert low_score <= high_score - 0.04
