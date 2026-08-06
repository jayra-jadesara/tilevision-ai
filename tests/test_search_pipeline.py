"""Unit tests for search pipeline scoring helpers."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

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


def test_marble_wood_stone_boost_handcrafted_signals():
    """Material tiles rely more on color/texture/edge than embedding alone."""
    for pattern_type in (PatternType.MARBLE, PatternType.WOOD, PatternType.STONE):
        weights = HybridReRanker.get_weights(pattern_type)
        handcrafted = (
            weights["color"] + weights["texture"] + weights["edge"] + weights["pattern"]
        )
        assert weights["embedding"] >= 0.50
        assert handcrafted >= 0.45
        assert weights["color"] + weights["texture"] >= 0.25


def test_marble_vs_wood_is_penalized():
    penalty = PatternClassifier.compatibility_adjustment(
        PatternType.MARBLE,
        PatternType.WOOD,
    )
    assert penalty <= -0.04


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


def _make_tile(tile_id: int, name: str, path: str) -> "TileImage":
    from src.core.models import TileImage

    return TileImage(
        file_path=path,
        file_name=name,
        file_size=1,
        dimensions="64x64",
        id=tile_id,
    )


def test_orb_near_tie_band_reorders_with_inlier_support(tmp_path, monkeypatch):
    """Near-identical hybrid scores: ORB inliers promote the true match."""
    import cv2
    from src.core.use_cases import search_tiles as st

    rng = np.random.default_rng(42)
    true_match = np.full((160, 160), 190, dtype=np.uint8)
    for _ in range(60):
        x0, y0 = int(rng.integers(0, 160)), int(rng.integers(0, 160))
        x1, y1 = int(rng.integers(0, 160)), int(rng.integers(0, 160))
        cv2.line(true_match, (x0, y0), (x1, y1), int(rng.integers(40, 160)), 2)

    distractor = rng.integers(0, 255, size=(160, 160), dtype=np.uint8)
    query = true_match.copy()

    true_path = tmp_path / "true.jpg"
    bad_path = tmp_path / "bad.jpg"
    Image.fromarray(true_match).save(true_path)
    Image.fromarray(distractor).save(bad_path)

    tile_true = _make_tile(1, "true.jpg", str(true_path))
    tile_bad = _make_tile(2, "bad.jpg", str(bad_path))

    # Deliberately close hybrid scores within the ORB band.
    reranked = [
        (0.72, tile_bad, False),
        (0.71, tile_true, False),
        (0.50, _make_tile(3, "far.jpg", str(bad_path)), False),
    ]

    usecase = object.__new__(st.SearchTilesUseCase)
    from src.ai.verification.orb_verifier import OrbVerifier

    usecase._orb_verifier = OrbVerifier()
    usecase._enable_orb_verification = True

    updated, applied = usecase._apply_orb_verification(reranked, query)
    assert applied >= 1
    assert updated[0][1].id == 1


def test_orb_skips_when_hybrid_scores_well_separated():
    """Regression: ORB must not fire when #1 is already clearly ahead."""
    from src.core.use_cases import search_tiles as st
    from src.ai.verification.orb_verifier import OrbVerifier

    tiles = [
        _make_tile(1, "a.jpg", "/tmp/a.jpg"),
        _make_tile(2, "b.jpg", "/tmp/b.jpg"),
        _make_tile(3, "c.jpg", "/tmp/c.jpg"),
    ]
    reranked = [
        (0.90, tiles[0], False),
        (0.70, tiles[1], False),  # gap 0.20 ≫ ORB_VERIFICATION_BAND
        (0.55, tiles[2], False),
    ]

    usecase = object.__new__(st.SearchTilesUseCase)
    usecase._orb_verifier = OrbVerifier()
    usecase._enable_orb_verification = True

    query = np.full((64, 64), 128, dtype=np.uint8)
    updated, applied = usecase._apply_orb_verification(reranked, query)
    assert applied == 0
    assert updated is reranked or [t.id for _, t, _ in updated] == [1, 2, 3]
    assert [t.id for _, t, _ in updated] == [1, 2, 3]

