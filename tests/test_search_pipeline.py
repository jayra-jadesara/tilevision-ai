"""Unit tests for search pipeline scoring helpers."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.descriptors.color_descriptor import ColorDescriptor
from src.ai.feature_versions import (
    CURRENT_FEATURE_VERSION,
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
            pattern or [0.0] * 8,
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
        pattern_feature_version=2,
        embedding_model="facebook/dinov2-large",
        embedding_dimension=1024,
        pattern_feature_size=8,
    )
    assert not is_tile_features_compatible(
        feature_version=1,
        pattern_feature_version=2,
        embedding_model="facebook/dinov2-large",
        embedding_dimension=1024,
        pattern_feature_size=8,
    )
    assert not is_tile_features_compatible(
        feature_version=2,
        pattern_feature_version=2,
        embedding_model="facebook/dinov2-large",
        embedding_dimension=1024,
        pattern_feature_size=8,
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


def test_identical_embeddings_score_highest():
    reranker = HybridReRanker()
    query = _features([1.0, 0.0, 0.0, 0.0])
    same = _features([1.0, 0.0, 0.0, 0.0])
    different = _features([0.0, 1.0, 0.0, 0.0])

    same_score = reranker.score(query, same).final
    diff_score = reranker.score(query, different).final

    assert same_score > diff_score
