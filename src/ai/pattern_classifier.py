from __future__ import annotations

from enum import Enum

import numpy as np

from src.ai.models import TileFeatures


class PatternType(str, Enum):
    SPECKLED = "speckled"
    TERRAZZO = "terrazzo"
    MARBLE = "marble"
    PLAIN = "plain"
    TEXTURED = "textured"


class PatternClassifier:
    """
    Rule-based tile pattern classifier.

    Current pattern_features format:
    [
        density,
        mean_size,
        size_std,
        count_normalized,
        coverage,
        size_consistency,
        spatial_uniformity,
        small_blob_ratio,
    ]

    The classifier intentionally uses conservative rules.

    SPECKLED:
        Many small, relatively consistent particles.

    TERRAZZO:
        Larger particles with more variation and coverage.

    PLAIN:
        Very few detected particles.

    MARBLE:
        Low/moderate particle activity with structural edge activity.

    TEXTURED:
        Fallback for surfaces that do not confidently match
        another pattern type.
    """

    @classmethod
    def classify(
        cls,
        features: TileFeatures,
    ) -> PatternType:

        pattern = features.pattern_features

        if pattern is None:
            return PatternType.TEXTURED

        pattern = np.asarray(
            pattern,
            dtype=np.float32,
        ).reshape(-1)

        if pattern.size < 5:
            return PatternType.TEXTURED

        density = float(pattern[0])
        mean_size = float(pattern[1])
        size_std = float(pattern[2])
        count_normalized = float(pattern[3])
        coverage = float(pattern[4])

        # Enhanced pattern features.
        # Defaults preserve compatibility with old 5-feature records.
        size_consistency = (
            float(pattern[5])
            if pattern.size >= 8
            else 0.0
        )

        spatial_uniformity = (
            float(pattern[6])
            if pattern.size >= 8
            else 0.0
        )

        small_blob_ratio = (
            float(pattern[7])
            if pattern.size >= 8
            else 0.0
        )

        print(
            "PATTERN FEATURES |",
            f"density={density:.6f} |",
            f"mean_size={mean_size:.6f} |",
            f"size_std={size_std:.6f} |",
            f"count={count_normalized:.4f} |",
            f"coverage={coverage:.4f} |",
            f"size_consistency={size_consistency:.4f} |",
            f"spatial_uniformity={spatial_uniformity:.4f} |",
            f"small_blob_ratio={small_blob_ratio:.4f}",
        )
        # ---------------------------------------------------------
        # Derived statistics
        # ---------------------------------------------------------

        # Measures how inconsistent blob sizes are.
        #
        # Speckles normally have relatively similar particle sizes.
        # Marble/noisy surfaces tend to produce blobs with much
        # larger size variation.
        if mean_size > 1e-8:
            size_variation = size_std / mean_size
        else:
            size_variation = 999.0

        # ---------------------------------------------------------
        # 1. Plain surface
        # ---------------------------------------------------------

        # Check this BEFORE speckled classification.
        #
        # A nearly plain surface may contain many tiny noise blobs
        # but still have extremely low total coverage.

        if (
            coverage < 0.008
            and density < 0.0015
        ):
            return PatternType.PLAIN

        # ---------------------------------------------------------
        # 2. Speckled / dotted
        # ---------------------------------------------------------

        # Speckled surfaces:
        # Many predominantly small particles distributed across the surface.
        #
        # Do not require high size_consistency here. Real speckled tiles can
        # contain a few merged/larger blobs that increase size_std even though
        # the overwhelming majority of detected particles are small.
        if (
            count_normalized >= 0.20
            and mean_size <= 0.0010
            and coverage >= 0.008
            and coverage <= 0.18
            and spatial_uniformity >= 0.40
            and small_blob_ratio >= 0.75
        ):
            return PatternType.SPECKLED

        # ---------------------------------------------------------
        # 3. Terrazzo
        # ---------------------------------------------------------

        # Terrazzo normally contains larger and/or more irregular
        # particles than fine speckled ceramic surfaces.

        if (
            coverage >= 0.04
            and (
                mean_size >= 0.0010
                or size_variation >= 2.5
            )
            and count_normalized >= 0.08
        ):
            return PatternType.TERRAZZO

        # ---------------------------------------------------------
        # 4. Marble
        # ---------------------------------------------------------

        # Marble should not be classified simply because an edge
        # histogram exists. Every indexed image normally has one.
        #
        # Instead, estimate actual edge activity.

        edge_hist = features.edge_histogram

        if edge_hist is not None:

            edge_hist = np.asarray(
                edge_hist,
                dtype=np.float32,
            ).reshape(-1)

            if edge_hist.size > 0:

                edge_activity = float(
                    np.mean(edge_hist)
                )

                if (
                    count_normalized < 0.25
                    and coverage < 0.12
                    and edge_activity > 0.02
                ):
                    return PatternType.MARBLE

        # ---------------------------------------------------------
        # 5. Plain fallback
        # ---------------------------------------------------------

        if (
            count_normalized < 0.08
            and coverage < 0.02
        ):
            return PatternType.PLAIN

        # ---------------------------------------------------------
        # 6. Generic texture
        # ---------------------------------------------------------

        return PatternType.TEXTURED