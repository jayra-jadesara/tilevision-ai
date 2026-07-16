from __future__ import annotations

import logging
from typing import List

from src.ai.descriptors.color_descriptor import ColorDescriptor
from src.ai.models import TileFeatures
from src.core.models import TileImage

logger = logging.getLogger("tilevision.ai.candidate_filter")


class CandidateFilter:
    """
    Broad pre-rerank color compatibility scorer.

    Applies a soft penalty for clearly incompatible dominant colors.
    Candidates are never hard-excluded — final ranking is handled by
    HybridReRanker, which calls ``dominant_color_penalty()``.
    """

    # LAB distance at which no penalty is applied.
    COLOR_DISTANCE_SOFT_START = 28.0
    # LAB distance at which the maximum penalty is reached.
    COLOR_DISTANCE_SOFT_END = 72.0
    # Maximum soft penalty applied to the hybrid final score.
    COLOR_PENALTY_MAX = 0.06

    @staticmethod
    def color_distance(c1, c2) -> float:
        return ColorDescriptor.rgb_to_lab_distance(
            tuple(c1),
            tuple(c2),
        )

    @classmethod
    def dominant_color_penalty(
        cls,
        query: TileFeatures,
        candidate: TileFeatures,
    ) -> float:
        """
        Return a soft penalty in [-COLOR_PENALTY_MAX, 0.0] based on dominant
        LAB color distance.  Same tile under different lighting stays near 0.
        """
        distance = cls.color_distance(
            query.dominant_color,
            candidate.dominant_color,
        )

        logger.debug(
            "Color compatibility | lab_distance=%.2f",
            distance,
        )

        if distance <= cls.COLOR_DISTANCE_SOFT_START:
            return 0.0
        if distance >= cls.COLOR_DISTANCE_SOFT_END:
            return -cls.COLOR_PENALTY_MAX

        ratio = (distance - cls.COLOR_DISTANCE_SOFT_START) / (
            cls.COLOR_DISTANCE_SOFT_END - cls.COLOR_DISTANCE_SOFT_START
        )
        return -cls.COLOR_PENALTY_MAX * ratio

    @classmethod
    def filter(
        cls,
        query: TileFeatures,
        candidates: List[TileImage],
    ) -> List[TileImage]:
        """
        Pass-through kept for backward compatibility.

        Color compatibility is now a soft reranker penalty, not a hard gate.
        """
        return [
            tile for tile in candidates if tile.features is not None
        ]
