"""
Hybrid reranker for TileVision AI.

Combines DINOv2 embedding similarity with handcrafted visual descriptors.
DINOv2 remains the dominant signal; handcrafted features provide fine-grained
refinement with bounded dynamic weighting and soft pattern-family compatibility.
"""

from __future__ import annotations

import numpy as np

from src.ai.models import TileFeatures, SearchScore
from src.ai.descriptors.color_descriptor import ColorDescriptor
from src.ai.descriptors.texture_descriptor import TextureDescriptor
from src.ai.descriptors.edge_descriptor import EdgeDescriptor
from src.ai.descriptors.pattern_descriptor import PatternDescriptor
from src.ai.pattern_classifier import PatternClassifier, PatternType

# DINOv2 must never fall below this fraction of the final blend.
_MIN_EMBEDDING_WEIGHT = 0.50

# Soft compatibility adjustment applied after the weighted blend.
# Bounded to avoid hard exclusions from classifier mistakes.
_COMPAT_BOOST_SAME = 0.02
_COMPAT_PENALTY_INCOMPATIBLE = -0.03


class HybridReRanker:

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        if a.size == 0 or b.size == 0:
            return 0.0
        if a.shape != b.shape:
            return 0.0

        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator < 1e-8:
            return 0.0

        return float(np.dot(a, b) / denominator)

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def pattern_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        a = np.asarray(a, dtype=np.float32).flatten()
        b = np.asarray(b, dtype=np.float32).flatten()
        if a.size == 0 or b.size == 0 or a.shape != b.shape:
            return 0.0
        return PatternDescriptor.similarity(a, b)

    def score(
        self,
        query: TileFeatures,
        candidate: TileFeatures,
        query_pattern_type: PatternType | None = None,
        candidate_pattern_type: PatternType | None = None,
    ) -> SearchScore:
        embedding = self.cosine_similarity(query.embedding, candidate.embedding)

        color = self._clamp01(
            ColorDescriptor.similarity(
                query.color_histogram,
                candidate.color_histogram,
            )
        )

        texture = self._clamp01(
            TextureDescriptor.similarity(
                query.texture_histogram,
                candidate.texture_histogram,
            )
        )

        edge = self._clamp01(
            EdgeDescriptor.similarity(
                query.edge_histogram,
                candidate.edge_histogram,
            )
        )

        pattern = self.pattern_similarity(
            query.pattern_features,
            candidate.pattern_features,
        )

        if query_pattern_type is None:
            query_pattern_type = PatternClassifier.classify(query)
        if candidate_pattern_type is None:
            candidate_pattern_type = PatternClassifier.classify(candidate)

        weights = self.get_weights(query_pattern_type)

        base_final = (
            embedding * weights["embedding"]
            + pattern * weights["pattern"]
            + color * weights["color"]
            + texture * weights["texture"]
            + edge * weights["edge"]
        )

        compat = PatternClassifier.compatibility_adjustment(
            query_pattern_type,
            candidate_pattern_type,
        )
        final = self._clamp01(base_final + compat)

        # Weak DINOv2 matches must not outrank strong semantic neighbors
        # because texture/color descriptors align on unrelated surfaces.
        if (
            query_pattern_type in (PatternType.SPECKLED, PatternType.TERRAZZO)
            and embedding < 0.42
        ):
            final *= 0.82

        return SearchScore(
            embedding=embedding,
            color=color,
            texture=texture,
            edge=edge,
            pattern=pattern,
            final=final,
        )

    @staticmethod
    def get_weights(pattern_type: PatternType) -> dict[str, float]:
        """
        Dynamic per-pattern weights.  DINOv2 embedding weight is always >= 0.50.
        """
        if pattern_type == PatternType.SPECKLED:
            weights = {
                "embedding": 0.70,
                "pattern": 0.20,
                "color": 0.05,
                "texture": 0.03,
                "edge": 0.02,
            }
        elif pattern_type == PatternType.TERRAZZO:
            weights = {
                "embedding": 0.55,
                "pattern": 0.20,
                "color": 0.12,
                "texture": 0.08,
                "edge": 0.05,
            }
        elif pattern_type == PatternType.MARBLE:
            weights = {
                "embedding": 0.60,
                "pattern": 0.03,
                "color": 0.10,
                "texture": 0.12,
                "edge": 0.15,
            }
        elif pattern_type == PatternType.WOOD:
            weights = {
                "embedding": 0.60,
                "pattern": 0.05,
                "color": 0.08,
                "texture": 0.12,
                "edge": 0.15,
            }
        elif pattern_type == PatternType.GEOMETRIC:
            weights = {
                "embedding": 0.65,
                "pattern": 0.08,
                "color": 0.08,
                "texture": 0.07,
                "edge": 0.12,
            }
        elif pattern_type == PatternType.STONE:
            weights = {
                "embedding": 0.58,
                "pattern": 0.12,
                "color": 0.12,
                "texture": 0.13,
                "edge": 0.05,
            }
        elif pattern_type == PatternType.MOSAIC:
            weights = {
                "embedding": 0.58,
                "pattern": 0.18,
                "color": 0.12,
                "texture": 0.07,
                "edge": 0.05,
            }
        elif pattern_type == PatternType.PLAIN:
            weights = {
                "embedding": 0.55,
                "pattern": 0.03,
                "color": 0.30,
                "texture": 0.07,
                "edge": 0.05,
            }
        else:
            # TEXTURED and fallback
            weights = {
                "embedding": 0.55,
                "pattern": 0.12,
                "color": 0.13,
                "texture": 0.13,
                "edge": 0.07,
            }

        # Safety: enforce DINOv2 as the strongest signal.
        if weights["embedding"] < _MIN_EMBEDDING_WEIGHT:
            deficit = _MIN_EMBEDDING_WEIGHT - weights["embedding"]
            weights["embedding"] = _MIN_EMBEDDING_WEIGHT
            others = ("pattern", "color", "texture", "edge")
            other_sum = sum(weights[k] for k in others)
            if other_sum > 0:
                for key in others:
                    weights[key] -= deficit * (weights[key] / other_sum)

        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            for key in weights:
                weights[key] /= total

        return weights
