from __future__ import annotations

import logging
from typing import List

import numpy as np

from src.ai.models import TileFeatures
from src.core.models import TileImage

logger = logging.getLogger("tilevision.ai.candidate_filter")


class CandidateFilter:
    """
    Broad pre-rerank filter.  Only rejects candidates with clearly
    incompatible dominant colors.  Final ranking is handled by HybridReRanker.
    """

    # Maximum RGB dominant-color Euclidean distance.
    # Tolerant of moderate lighting / white-balance differences.
    COLOR_DISTANCE_THRESHOLD = 90.0

    @staticmethod
    def color_distance(c1, c2) -> float:
        c1 = np.asarray(c1, dtype=np.float32)
        c2 = np.asarray(c2, dtype=np.float32)
        return float(np.linalg.norm(c1 - c2))

    @classmethod
    def filter(
        cls,
        query: TileFeatures,
        candidates: List[TileImage],
    ) -> List[TileImage]:

        filtered: List[TileImage] = []

        for tile in candidates:
            features = tile.features
            if features is None:
                continue

            color_distance = cls.color_distance(
                query.dominant_color,
                features.dominant_color,
            )

            logger.debug(
                "Candidate filter | %s | color_distance=%.2f",
                tile.file_name,
                color_distance,
            )

            if color_distance > cls.COLOR_DISTANCE_THRESHOLD:
                continue

            filtered.append(tile)

        return filtered
