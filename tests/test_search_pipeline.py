"""Unit tests for search pipeline scoring helpers."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.models import TileFeatures
from src.ai.descriptors.color_descriptor import ColorDescriptor
from src.ai.feature_versions import (
    CURRENT_FEATURE_VERSION,
    CURRENT_PATTERN_FEATURE_VERSION,
    is_tile_features_compatible,
)
from src.ai.pattern_classifier import PatternClassifier, PatternType
from src.ai.reranker import HybridReRanker
from src.ai.similarity_score import calibrate_display_percent
from src.ai.models import TileFeatures


def _features(
    embedding: list[float],
    pattern: list[float] | None = None,
) -> TileFeatures:
    return TileFeatures(
        embedding=np.asarray(embedding, dtype=np.float32),
        color_histogram=np.full(
            ColorDescriptor.vector_size(),
            1.0 / ColorDescriptor.vector_size(),
            dtype=np.float32,
        ),
        texture_histogram=np.full(54, 1.0 / 54, dtype=np.float32),
        edge_histogram=np.full(36, 1.0 / 36, dtype=np.float32),
        pattern_features=np.asarray(
            pattern or [0.0] * 12,
            dtype=np.float32,
        ),
        dominant_color=(200, 200, 200),
        width=32,
        height=32,
    )


def test_calibrate_exact_match_returns_100():
    assert calibrate_display_percent(0.5, exact_match=True) == 100.0


def test_calibrate_weak_match_is_compressed():
    weak = calibrate_display_percent(0.30)
    strong = calibrate_display_percent(0.85)
    assert weak < 30.0
    assert strong > weak
    assert strong < 99.5


def test_reranker_embedding_weight_is_at_least_half():
    for pattern_type in PatternType:
        weights = HybridReRanker.get_weights(pattern_type)
        assert weights["embedding"] >= 0.50
        assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_speckled_weights_favor_embedding():
    weights = HybridReRanker.get_weights(PatternType.SPECKLED)
    assert weights["embedding"] >= 0.65


def test_pattern_compatibility_penalizes_speckled_vs_plain():
    penalty = PatternClassifier.compatibility_adjustment(
        PatternType.SPECKLED,
        PatternType.PLAIN,
    )
    assert penalty < 0.0


def test_pattern_compatibility_boosts_same_family():
    boost = PatternClassifier.compatibility_adjustment(
        PatternType.MARBLE,
        PatternType.MARBLE,
    )
    assert boost > 0.0


def test_feature_version_detects_stale_records():
    assert is_tile_features_compatible(
        feature_version=CURRENT_FEATURE_VERSION,
        pattern_feature_version=CURRENT_PATTERN_FEATURE_VERSION,
        embedding_model="facebook/dinov2-large",
        embedding_dimension=1024,
        pattern_feature_size=12,
    )
    assert not is_tile_features_compatible(
        feature_version=1,
        pattern_feature_version=CURRENT_PATTERN_FEATURE_VERSION,
        embedding_model="facebook/dinov2-large",
        embedding_dimension=1024,
        pattern_feature_size=12,
    )
    assert not is_tile_features_compatible(
        feature_version=2,
        pattern_feature_version=CURRENT_PATTERN_FEATURE_VERSION,
        embedding_model="facebook/dinov2-large",
        embedding_dimension=1024,
        pattern_feature_size=12,
        color_histogram_size=ColorDescriptor.vector_size(),
    )
    assert not is_tile_features_compatible(
        feature_version=CURRENT_FEATURE_VERSION,
        pattern_feature_version=2,
        embedding_model="facebook/dinov2-large",
        embedding_dimension=1024,
        pattern_feature_size=8,
        color_histogram_size=8192,
    )
    assert not is_tile_features_compatible(
        feature_version=CURRENT_FEATURE_VERSION,
        pattern_feature_version=2,
        embedding_model="facebook/dinov2-large",
        embedding_dimension=1024,
        pattern_feature_size=8,
    )


def test_identical_embeddings_score_highest():
    reranker = HybridReRanker()
    query = _features([1.0, 0.0, 0.0, 0.0])
    same = _features([1.0, 0.0, 0.0, 0.0])
    different = _features([0.0, 1.0, 0.0, 0.0])

    same_score = reranker.score(query, same).final
    diff_score = reranker.score(query, different).final

    assert same_score > diff_score


def test_speckled_query_prefers_higher_embedding_over_texture_color():
    """Regression: cream marble must not outrank a closer white speckled match."""
    reranker = HybridReRanker()
    query = _features([0.65, 0.10, 0.05, 0.05])

    white_speckled = TileFeatures(
        embedding=np.asarray([0.65, 0.10, 0.05, 0.05], dtype=np.float32),
        color_histogram=np.full(ColorDescriptor.vector_size(), 0.01, dtype=np.float32),
        texture_histogram=np.full(54, 0.02, dtype=np.float32),
        edge_histogram=np.full(36, 0.02, dtype=np.float32),
        pattern_features=np.asarray(
            [0.003, 0.0006, 0.0006, 0.79, 0.046, 0.59, 0.86, 0.98],
            dtype=np.float32,
        ),
        dominant_color=(240, 240, 240),
        width=32,
        height=32,
    )
    cream_marble = TileFeatures(
        embedding=np.asarray([0.35, 0.20, 0.10, 0.05], dtype=np.float32),
        color_histogram=np.full(ColorDescriptor.vector_size(), 0.05, dtype=np.float32),
        texture_histogram=np.full(54, 0.08, dtype=np.float32),
        edge_histogram=np.full(36, 0.08, dtype=np.float32),
        pattern_features=np.asarray(
            [0.002, 0.0005, 0.0015, 0.31, 0.077, 0.24, 0.54, 0.81],
            dtype=np.float32,
        ),
        dominant_color=(220, 210, 190),
        width=32,
        height=32,
    )

    speckled_score = reranker.score(
        query,
        white_speckled,
        query_pattern_type=PatternType.SPECKLED,
        candidate_pattern_type=PatternType.SPECKLED,
    ).final
    cream_score = reranker.score(
        query,
        cream_marble,
        query_pattern_type=PatternType.SPECKLED,
        candidate_pattern_type=PatternType.SPECKLED,
    ).final

    assert speckled_score > cream_score
