"""
Color descriptor for TileVision AI.

Uses a LAB color histogram plus compact brightness/saturation statistics.
LAB separates perceptual color from lighting more robustly than raw RGB,
which improves matching under different photography conditions.

Near-white / low-chroma tiles are a special case: camera white-balance and
JPEG re-encoding shift a*/b* enough to collapse ``HISTCMP_CORREL`` on the
full LAB histogram (measured ~0.001 on the same marble under cool vs warm
WB) while a human still sees "the same white marble". Similarity and the
dominant-color penalty therefore fall back to lightness/chroma agreement
when both sides are near-white.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger("tilevision.ai.color_descriptor")

# LAB histogram bins (OpenCV LAB ranges are 0-255 per channel).
L_BINS = 20
A_BINS = 12
B_BINS = 12

# Trailing stats: mean_L, std_L, mean_saturation, mean_value (all normalized).
STATS_SIZE = 4

HISTOGRAM_SIZE = L_BINS * A_BINS * B_BINS
VECTOR_SIZE = HISTOGRAM_SIZE + STATS_SIZE

# Normalized HSV/LAB stats thresholds for the near-white soft path.
_NEAR_WHITE_MEAN_L = 0.70
_NEAR_WHITE_MAX_SAT = 0.14
# OpenCV LAB L is 0–255; a/b centered near 128.
_NEAR_WHITE_DOM_L = 175.0
_NEAR_WHITE_DOM_CHROMA = 28.0


class ColorDescriptor:
    """LAB histogram + global color statistics."""

    @classmethod
    def vector_size(cls) -> int:
        return VECTOR_SIZE

    @classmethod
    def extract(
        cls,
        image_bgr: np.ndarray,
    ) -> np.ndarray:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        histogram = cv2.calcHist(
            [lab],
            [0, 1, 2],
            None,
            [L_BINS, A_BINS, B_BINS],
            [0, 256, 0, 256, 0, 256],
        )
        histogram = histogram.astype(np.float32).flatten()
        cv2.normalize(histogram, histogram, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L2)

        l_channel = lab[:, :, 0].astype(np.float32)
        saturation = hsv[:, :, 1].astype(np.float32)
        value = hsv[:, :, 2].astype(np.float32)

        stats = np.asarray(
            [
                float(l_channel.mean() / 255.0),
                float(l_channel.std() / 255.0),
                float(saturation.mean() / 255.0),
                float(value.mean() / 255.0),
            ],
            dtype=np.float32,
        )

        return np.concatenate([histogram, stats]).astype(np.float32)

    @staticmethod
    def _stats_near_white(stats: np.ndarray) -> bool:
        mean_l = float(stats[0])
        sat = float(stats[2])
        return mean_l >= _NEAR_WHITE_MEAN_L and sat <= _NEAR_WHITE_MAX_SAT

    @staticmethod
    def _near_white_similarity(query_stats: np.ndarray, cand_stats: np.ndarray) -> float:
        """
        Lightness/chroma agreement for near-white pairs.

        Ignores a*/b* bin shifts that destroy full LAB histogram correlation
        under camera white-balance changes.
        """
        l_delta = abs(float(query_stats[0]) - float(cand_stats[0]))
        sat_delta = abs(float(query_stats[2]) - float(cand_stats[2]))
        val_delta = abs(float(query_stats[3]) - float(cand_stats[3]))
        l_sim = max(0.0, 1.0 - l_delta / 0.28)
        sat_sim = max(0.0, 1.0 - sat_delta / 0.20)
        val_sim = max(0.0, 1.0 - val_delta / 0.28)
        return float(np.clip(0.55 * l_sim + 0.25 * val_sim + 0.20 * sat_sim, 0.0, 1.0))

    @staticmethod
    def similarity(
        query_hist: np.ndarray,
        candidate_hist: np.ndarray,
    ) -> float:
        query_hist = np.asarray(query_hist, dtype=np.float32).reshape(-1)
        candidate_hist = np.asarray(candidate_hist, dtype=np.float32).reshape(-1)

        if query_hist.size != candidate_hist.size:
            return 0.0

        if query_hist.size == VECTOR_SIZE:
            query_core = query_hist[:HISTOGRAM_SIZE]
            candidate_core = candidate_hist[:HISTOGRAM_SIZE]
            query_stats = query_hist[HISTOGRAM_SIZE:]
            candidate_stats = candidate_hist[HISTOGRAM_SIZE:]

            hist_score = float(
                cv2.compareHist(
                    query_core,
                    candidate_core,
                    cv2.HISTCMP_CORREL,
                )
            )

            stat_distance = float(
                np.linalg.norm(query_stats - candidate_stats)
            )
            stat_score = max(0.0, 1.0 - stat_distance)

            base = max(
                0.0,
                min(1.0, 0.80 * hist_score + 0.20 * stat_score),
            )

            if ColorDescriptor._stats_near_white(query_stats) and (
                ColorDescriptor._stats_near_white(candidate_stats)
            ):
                white_sim = ColorDescriptor._near_white_similarity(
                    query_stats,
                    candidate_stats,
                )
                # Prefer the better of full LAB hist vs WB-robust near-white path.
                return float(max(base, white_sim))

            return base

        # Backward compatibility with legacy HSV histograms (8192 bins).
        score = cv2.compareHist(query_hist, candidate_hist, cv2.HISTCMP_CORREL)
        return max(0.0, min(1.0, float(score)))

    @staticmethod
    def rgb_to_lab_distance(
        color_a: tuple[int, int, int],
        color_b: tuple[int, int, int],
    ) -> float:
        """
        Perceptual color distance in LAB space.

        Parameters are RGB tuples as stored in TileFeatures.dominant_color.
        """
        a = np.uint8([[[color_a[0], color_a[1], color_a[2]]]])
        b = np.uint8([[[color_b[0], color_b[1], color_b[2]]]])
        lab_a = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_b = cv2.cvtColor(b, cv2.COLOR_RGB2LAB).astype(np.float32)
        return float(np.linalg.norm(lab_a - lab_b))

    @staticmethod
    def is_near_white_rgb(color_rgb: tuple[int, int, int]) -> bool:
        """True when a dominant RGB color is high-L / low-chroma in OpenCV LAB."""
        arr = np.uint8([[[color_rgb[0], color_rgb[1], color_rgb[2]]]])
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
        l_val = float(lab[0])
        chroma = float(np.hypot(lab[1] - 128.0, lab[2] - 128.0))
        return l_val >= _NEAR_WHITE_DOM_L and chroma <= _NEAR_WHITE_DOM_CHROMA

    @staticmethod
    def dominant_color_rgb(image_bgr: np.ndarray) -> tuple[int, int, int]:
        """
        Estimate dominant color using LAB k-means, returned as an RGB tuple.
        """
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape((-1, 3)).astype(np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20,
            1.0,
        )
        _, _, centers = cv2.kmeans(
            pixels,
            1,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS,
        )

        center_lab = np.uint8([[centers[0].astype(np.uint8)]])
        center_bgr = cv2.cvtColor(center_lab, cv2.COLOR_LAB2BGR)[0, 0]
        return (
            int(center_bgr[2]),
            int(center_bgr[1]),
            int(center_bgr[0]),
        )

    @staticmethod
    def serialize(histogram: np.ndarray) -> bytes:
        return histogram.astype(np.float32).tobytes()

    @staticmethod
    def deserialize(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)
