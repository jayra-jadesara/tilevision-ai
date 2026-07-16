from __future__ import annotations

import logging
from enum import Enum

import numpy as np

from src.ai.models import TileFeatures

logger = logging.getLogger("tilevision.ai.pattern_classifier")


class PatternType(str, Enum):
    SPECKLED = "speckled"
    TERRAZZO = "terrazzo"
    MARBLE = "marble"
    PLAIN = "plain"
    TEXTURED = "textured"


class PatternClassifier:
    """
    Conservative rule-based tile pattern classifier.

    Classification is intentionally soft at the boundaries — ambiguous
    surfaces fall back to TEXTURED rather than forcing a wrong family.
    """

    @staticmethod
    def _parse_pattern_features(pattern: np.ndarray | None) -> dict[str, float] | None:
        if pattern is None:
            return None

        pattern = np.asarray(pattern, dtype=np.float32).reshape(-1)
        if pattern.size < 5:
            return None

        values = {
            "density": float(pattern[0]),
            "mean_size": float(pattern[1]),
            "size_std": float(pattern[2]),
            "count_normalized": float(pattern[3]),
            "coverage": float(pattern[4]),
            "size_consistency": float(pattern[5]) if pattern.size >= 8 else 0.0,
            "spatial_uniformity": float(pattern[6]) if pattern.size >= 8 else 0.0,
            "small_blob_ratio": float(pattern[7]) if pattern.size >= 8 else 0.0,
        }

        mean_size = values["mean_size"]
        values["size_variation"] = (
            values["size_std"] / mean_size if mean_size > 1e-8 else 999.0
        )
        return values

    @staticmethod
    def _edge_metrics(edge_hist: np.ndarray | None) -> tuple[float, float, float]:
        if edge_hist is None:
            return 0.0, 0.0, 0.0

        hist = np.asarray(edge_hist, dtype=np.float32).reshape(-1)
        if hist.size == 0:
            return 0.0, 0.0, 0.0

        activity = float(np.mean(hist))
        total = float(hist.sum()) + 1e-8
        normalized = hist / total
        directionality = float(normalized.max() * hist.size)
        entropy = float(
            -np.sum(normalized * np.log(normalized + 1e-8))
        )
        return activity, directionality, entropy

    @classmethod
    def _score_families(
        cls,
        values: dict[str, float],
        edge_activity: float,
        edge_directionality: float,
        edge_entropy: float,
    ) -> dict[PatternType, float]:
        """
        Compute soft family scores in [0, 1].  Highest score wins only if
        it clears a minimum confidence gap over the runner-up.
        """
        density = values["density"]
        mean_size = values["mean_size"]
        count_normalized = values["count_normalized"]
        coverage = values["coverage"]
        spatial_uniformity = values["spatial_uniformity"]
        small_blob_ratio = values["small_blob_ratio"]
        size_variation = values["size_variation"]

        scores = {
            PatternType.PLAIN: 0.0,
            PatternType.SPECKLED: 0.0,
            PatternType.TERRAZZO: 0.0,
            PatternType.MARBLE: 0.0,
            PatternType.TEXTURED: 0.15,
        }

        # Plain: very low particle activity.
        if coverage < 0.012 and count_normalized < 0.10:
            scores[PatternType.PLAIN] = max(
                scores[PatternType.PLAIN],
                0.55 + (0.012 - coverage) * 10.0,
            )

        # Marble: directional structure with few small particles.
        # Cream marble often fails old speckled rules because of fine noise,
        # but still has vein-like edges and low small-blob dominance.
        marble_signal = 0.0
        if small_blob_ratio < 0.60:
            marble_signal += min(0.35, edge_activity * 8.0)
            marble_signal += min(0.25, max(0.0, edge_directionality - 2.0) * 0.12)
            if count_normalized < 0.20:
                marble_signal += 0.15
            if coverage < 0.12:
                marble_signal += 0.10
            if edge_entropy > 2.0:
                marble_signal += 0.05
            if small_blob_ratio > 0.45:
                marble_signal *= 0.60
            if 0.35 <= small_blob_ratio <= 0.55:
                marble_signal *= 0.55
        scores[PatternType.MARBLE] = min(1.0, marble_signal)

        # Speckled: many small particles spread across the surface.
        speckled_signal = 0.0
        if small_blob_ratio >= 0.60:
            speckled_signal += 0.25
        if count_normalized >= 0.12:
            speckled_signal += min(0.25, count_normalized)
        if 0.006 <= coverage <= 0.22:
            speckled_signal += 0.15
        if mean_size <= 0.0012:
            speckled_signal += 0.10
        if spatial_uniformity >= 0.35:
            speckled_signal += 0.10
        if edge_directionality < 4.0:
            speckled_signal += 0.10
        scores[PatternType.SPECKLED] = min(1.0, speckled_signal)

        # Terrazzo: broader, more irregular particles than fine speckles.
        terrazzo_signal = 0.0
        if coverage >= 0.035:
            terrazzo_signal += 0.20
        if count_normalized >= 0.06:
            terrazzo_signal += 0.10
        if mean_size >= 0.0009 or size_variation >= 2.0:
            terrazzo_signal += 0.20
        if small_blob_ratio < 0.70:
            terrazzo_signal += 0.10
        if size_variation >= 2.5:
            terrazzo_signal += 0.10
        scores[PatternType.TERRAZZO] = min(1.0, terrazzo_signal)

        # Penalize contradictory combinations.
        if scores[PatternType.MARBLE] > 0.45 and scores[PatternType.SPECKLED] > 0.45:
            if small_blob_ratio >= 0.65:
                scores[PatternType.MARBLE] *= 0.55
            else:
                scores[PatternType.SPECKLED] *= 0.55

        if density < 0.0010 and coverage < 0.01:
            scores[PatternType.PLAIN] = max(scores[PatternType.PLAIN], 0.70)

        return scores

    @classmethod
    def classify(
        cls,
        features: TileFeatures,
    ) -> PatternType:
        values = cls._parse_pattern_features(features.pattern_features)
        if values is None:
            return PatternType.TEXTURED

        edge_activity, edge_directionality, edge_entropy = cls._edge_metrics(
            features.edge_histogram
        )

        logger.debug(
            "Pattern features: density=%.6f mean_size=%.6f size_std=%.6f "
            "count=%.4f coverage=%.4f size_consistency=%.4f "
            "spatial_uniformity=%.4f small_blob_ratio=%.4f edge_activity=%.4f "
            "edge_directionality=%.2f edge_entropy=%.2f",
            values["density"],
            values["mean_size"],
            values["size_std"],
            values["count_normalized"],
            values["coverage"],
            values["size_consistency"],
            values["spatial_uniformity"],
            values["small_blob_ratio"],
            edge_activity,
            edge_directionality,
            edge_entropy,
        )

        scores = cls._score_families(
            values,
            edge_activity,
            edge_directionality,
            edge_entropy,
        )

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_type, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        # Middle blob-ratio band is inherently ambiguous (neither veiny marble
        # nor dense fine speckles) — demand a stronger margin before committing.
        small_blob_ratio = values["small_blob_ratio"]
        min_gap = 0.08
        if 0.35 <= small_blob_ratio <= 0.55:
            min_gap = 0.15

        # Require a clear winner; otherwise stay generic.
        if best_score < 0.40 or (best_score - second_score) < min_gap:
            return PatternType.TEXTURED

        logger.debug(
            "Pattern classification: %s (score=%.3f, runner_up=%.3f)",
            best_type.value,
            best_score,
            second_score,
        )
        return best_type

    @classmethod
    def classify_scores(
        cls,
        features: TileFeatures,
    ) -> dict[PatternType, float]:
        """Return soft family scores for diagnostics or future reranking."""
        values = cls._parse_pattern_features(features.pattern_features)
        if values is None:
            return {PatternType.TEXTURED: 1.0}

        edge_activity, edge_directionality, edge_entropy = cls._edge_metrics(
            features.edge_histogram
        )
        return cls._score_families(
            values,
            edge_activity,
            edge_directionality,
            edge_entropy,
        )

    @staticmethod
    def compatibility_adjustment(
        query_type: PatternType,
        candidate_type: PatternType,
    ) -> float:
        """
        Soft ranking signal based on pattern-family compatibility.

        Returns a small additive adjustment in [-0.03, +0.02].
        Never hard-excludes candidates.
        """
        if query_type == candidate_type:
            return 0.02

        pair = frozenset({query_type, candidate_type})

        compatible_pairs = {
            frozenset({PatternType.SPECKLED, PatternType.TERRAZZO}),
            frozenset({PatternType.MARBLE, PatternType.TEXTURED}),
            frozenset({PatternType.PLAIN, PatternType.TEXTURED}),
            frozenset({PatternType.MARBLE, PatternType.PLAIN}),
        }
        if pair in compatible_pairs:
            return 0.0

        incompatible_pairs = {
            frozenset({PatternType.SPECKLED, PatternType.PLAIN}),
            frozenset({PatternType.SPECKLED, PatternType.MARBLE}),
            frozenset({PatternType.PLAIN, PatternType.TERRAZZO}),
        }
        if pair in incompatible_pairs:
            return -0.03

        return 0.0
