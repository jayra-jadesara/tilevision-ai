from __future__ import annotations

from typing import List

import cv2
import numpy as np

from src.ai.models import TileFeatures
from src.core.models import TileImage


class CandidateFilter:
    """
    Removes visually incompatible candidates before hybrid reranking.

    This is intentionally a broad filter. The final ranking is handled
    by HybridReRanker.
    """

    # Maximum RGB dominant-color Euclidean distance.
    COLOR_DISTANCE_THRESHOLD = 75.0

    # Minimum histogram correlation.
    # OpenCV correlation range is approximately -1 to 1.
    COLOR_HISTOGRAM_THRESHOLD = 0.05

    @staticmethod
    def color_distance(c1, c2) -> float:
        c1 = np.asarray(c1, dtype=np.float32)
        c2 = np.asarray(c2, dtype=np.float32)

        return float(
            np.linalg.norm(c1 - c2)
        )

    @staticmethod
    def histogram_similarity(
        query_hist: np.ndarray,
        candidate_hist: np.ndarray,
    ) -> float:

        if query_hist is None or candidate_hist is None:
            return 0.0

        query_hist = np.asarray(
            query_hist,
            dtype=np.float32,
        ).reshape(-1)

        candidate_hist = np.asarray(
            candidate_hist,
            dtype=np.float32,
        ).reshape(-1)

        if query_hist.size == 0 or candidate_hist.size == 0:
            return 0.0

        if query_hist.size != candidate_hist.size:
            return 0.0

        try:
            print(
                "HIST DEBUG:",
                "query_shape=", query_hist.shape,
                "candidate_shape=", candidate_hist.shape,
                "query_size=", query_hist.size,
                "candidate_size=", candidate_hist.size,
                "query_dtype=", query_hist.dtype,
                "candidate_dtype=", candidate_hist.dtype,
            )

            return float(
                cv2.compareHist(
                    query_hist,
                    candidate_hist,
                    cv2.HISTCMP_CORREL,
                )
            )
        except cv2.error as e:
            print(f"HIST ERROR: {e}")
            return 0.0

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

            # -----------------------------------------
            # Dominant color check only
            # -----------------------------------------

            color_distance = cls.color_distance(
                query.dominant_color,
                features.dominant_color,
            )

            print(
                f"FILTER | {tile.file_name} | "
                f"color_distance={color_distance:.2f}"
            )

            if color_distance > cls.COLOR_DISTANCE_THRESHOLD:
                continue

            filtered.append(tile)

        return filtered