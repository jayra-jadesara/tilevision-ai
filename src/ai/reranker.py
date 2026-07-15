"""
Hybrid reranker for TileVision AI.

Combines DINOv2 embedding similarity with handcrafted
visual descriptors and pattern features.
"""

from __future__ import annotations

import numpy as np

from src.ai.models import TileFeatures, SearchScore
from src.ai.descriptors.pattern_descriptor import PatternDescriptor

from src.ai.pattern_classifier import (
    PatternClassifier,
    PatternType,
)

class HybridReRanker:

    # ---------------------------------------------------------
    # Weight configuration
    # ---------------------------------------------------------
    #
    # For ceramic tile visual search:
    #
    # DINOv2  -> overall visual appearance
    # Pattern -> dots / speckles / repeated local patterns
    # Color   -> overall color distribution
    # Texture -> surface texture
    # Edge    -> structural lines / boundaries
    #
    # Total = 1.0

    @staticmethod
    def cosine_similarity(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:

        if a is None or b is None:
            return 0.0

        if a.size == 0 or b.size == 0:
            return 0.0

        if a.shape != b.shape:
            return 0.0

        denominator = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        if denominator < 1e-8:
            return 0.0

        return float(
            np.dot(a, b)
            / denominator
        )

    @staticmethod
    def histogram_similarity(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:

        if a is None or b is None:
            return 0.0

        a = np.asarray(
            a,
            dtype=np.float32,
        ).reshape(-1)

        b = np.asarray(
            b,
            dtype=np.float32,
        ).reshape(-1)

        if a.size == 0 or b.size == 0:
            return 0.0

        if a.size != b.size:
            return 0.0

        # Avoid undefined correlation for constant arrays
        a_std = float(np.std(a))
        b_std = float(np.std(b))

        if a_std < 1e-8 or b_std < 1e-8:
            return 1.0 if np.allclose(a, b) else 0.0

        score = np.corrcoef(a, b)[0, 1]

        if not np.isfinite(score):
            return 0.0

        # Correlation range is -1 to 1.
        # For weighted similarity, clamp negative values to zero.
        return max(
            0.0,
            min(1.0, float(score)),
        )

    @staticmethod
    def pattern_similarity(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:

        if a is None or b is None:
            return 0.0

        a = np.asarray(
            a,
            dtype=np.float32,
        ).flatten()

        b = np.asarray(
            b,
            dtype=np.float32,
        ).flatten()

        if a.size == 0 or b.size == 0:
            return 0.0

        if a.shape != b.shape:
            return 0.0

        return PatternDescriptor.similarity(
            a,
            b,
        )

    def score(
        self,
        query: TileFeatures,
        candidate: TileFeatures,
    ) -> SearchScore:

        embedding = self.cosine_similarity(
            query.embedding,
            candidate.embedding,
        )

        color = self.histogram_similarity(
            query.color_histogram,
            candidate.color_histogram,
        )

        texture = self.histogram_similarity(
            query.texture_histogram,
            candidate.texture_histogram,
        )

        edge = self.histogram_similarity(
            query.edge_histogram,
            candidate.edge_histogram,
        )

        pattern = self.pattern_similarity(
            query.pattern_features,
            candidate.pattern_features,
        )

        pattern_type = PatternClassifier.classify(
            query
        )

        weights = self.get_weights(
            pattern_type
        )

        final = (
            embedding * weights["embedding"]
            + pattern * weights["pattern"]
            + color * weights["color"]
            + texture * weights["texture"]
            + edge * weights["edge"]
        )

        return SearchScore(
            embedding=embedding,
            color=color,
            texture=texture,
            edge=edge,
            pattern=pattern,
            final=final,
        )
    
    @staticmethod
    def get_weights(
        pattern_type: PatternType,
    ) -> dict[str, float]:

        if pattern_type == PatternType.SPECKLED:
            return {
                "embedding": 0.40,
                "pattern": 0.40,
                "color": 0.10,
                "texture": 0.07,
                "edge": 0.03,
            }

        if pattern_type == PatternType.TERRAZZO:
            return {
                "embedding": 0.40,
                "pattern": 0.35,
                "color": 0.12,
                "texture": 0.08,
                "edge": 0.05,
            }

        if pattern_type == PatternType.MARBLE:
            return {
                "embedding": 0.50,
                "pattern": 0.05,
                "color": 0.10,
                "texture": 0.15,
                "edge": 0.20,
            }

        if pattern_type == PatternType.PLAIN:
            return {
                "embedding": 0.50,
                "pattern": 0.05,
                "color": 0.30,
                "texture": 0.10,
                "edge": 0.05,
            }

        # Generic textured tile
        return {
            "embedding": 0.50,
            "pattern": 0.20,
            "color": 0.10,
            "texture": 0.15,
            "edge": 0.05,
        }