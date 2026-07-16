"""Tests for pattern classification logic."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.models import TileFeatures
from src.ai.pattern_classifier import PatternClassifier, PatternType


def _features(
    pattern: list[float],
    edge_activity: float = 0.01,
    directional: bool = False,
) -> TileFeatures:
    edge = np.zeros(36, dtype=np.float32)
    edge[:] = edge_activity / 36.0
    if directional:
        edge[4] = edge_activity
        edge[5] = edge_activity * 0.8

    return TileFeatures(
        embedding=np.ones(4, dtype=np.float32),
        color_histogram=np.ones(16, dtype=np.float32),
        texture_histogram=np.ones(16, dtype=np.float32),
        edge_histogram=edge,
        pattern_features=np.asarray(pattern, dtype=np.float32),
        dominant_color=(200, 200, 200),
        width=32,
        height=32,
    )


def test_dense_speckles_classified_as_speckled():
    features = _features(
        [
            0.0020,  # density
            0.0006,  # mean_size
            0.0002,  # size_std
            0.35,    # count_normalized
            0.08,    # coverage
            0.70,    # size_consistency
            0.55,    # spatial_uniformity
            0.82,    # small_blob_ratio
        ],
        edge_activity=0.015,
        directional=False,
    )
    assert PatternClassifier.classify(features) == PatternType.SPECKLED


def test_veiny_low_particle_surface_classified_as_marble_not_speckled():
    features = _features(
        [
            0.0008,
            0.0005,
            0.0004,
            0.10,
            0.05,
            0.40,
            0.45,
            0.20,
        ],
        edge_activity=0.06,
        directional=True,
    )
    result = PatternClassifier.classify(features)
    assert result == PatternType.MARBLE


def test_low_activity_surface_classified_as_plain():
    features = _features(
        [
            0.0002,
            0.0001,
            0.0001,
            0.02,
            0.004,
            0.50,
            0.30,
            0.10,
        ],
        edge_activity=0.005,
    )
    assert PatternClassifier.classify(features) == PatternType.PLAIN


def test_ambiguous_surface_falls_back_to_textured():
    features = _features(
        [
            0.0010,
            0.0008,
            0.0006,
            0.11,
            0.04,
            0.42,
            0.38,
            0.42,
        ],
        edge_activity=0.028,
        directional=True,
    )
    assert PatternClassifier.classify(features) == PatternType.TEXTURED


def test_marble_and_speckled_incompatible_penalty():
    penalty = PatternClassifier.compatibility_adjustment(
        PatternType.MARBLE,
        PatternType.SPECKLED,
    )
    assert penalty < 0.0
