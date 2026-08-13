"""
Edge descriptor for TileVision AI.

Extracts edge-orientation features using Canny + Sobel gradients.

Purpose
-------
DINO learns semantic similarity.

LBP learns texture.

This descriptor learns structural information such as

- marble veins
- wood grain
- stone cracks
- tile pattern direction
- geometric layouts

Author:
TileVision AI v2
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger("tilevision.ai.edge_descriptor")

# Cosine denom / empty-hist epsilon.
_EPS = 1e-8
# Adaptive Canny: derive hysteresis from the image's own Sobel magnitude
# distribution. Fixed 80/180 (legacy) found zero edges on correctly exposed
# cream marble — the same low-contrast class that broke normalize_lighting.
_CANNY_MAG_PERCENTILE = 92.0
_CANNY_HIGH_FLOOR = 12.0
_CANNY_HIGH_CEIL = 200.0
_CANNY_LOW_RATIO = 0.4
# If adaptive Canny is still nearly empty, fall back to a magnitude mask.
_MIN_CANNY_DENSITY = 0.002


class EdgeDescriptor:
    """
    Edge orientation descriptor.
    """

    ORIENTATION_BINS = 36

    @classmethod
    def _adaptive_canny_thresholds(cls, magnitude: np.ndarray) -> tuple[int, int]:
        """Hysteresis from this frame's gradient stats, not fixed constants."""
        high = float(np.percentile(magnitude, _CANNY_MAG_PERCENTILE))
        high = float(np.clip(high, _CANNY_HIGH_FLOOR, _CANNY_HIGH_CEIL))
        low = max(high * _CANNY_LOW_RATIO, 1.0)
        return int(round(low)), int(round(high))

    @classmethod
    def extract(
        cls,
        image_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Extract edge orientation histogram.

        Parameters
        ----------
        image_bgr
            OpenCV BGR image.

        Returns
        -------
        ndarray
            float32 normalized histogram
        """

        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        gx = cv2.Sobel(
            gray,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )

        gy = cv2.Sobel(
            gray,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
        )

        magnitude, angle = cv2.cartToPolar(
            gx,
            gy,
            angleInDegrees=True,
        )

        low, high = cls._adaptive_canny_thresholds(magnitude)
        edges = cv2.Canny(
            gray,
            threshold1=low,
            threshold2=high,
        )

        edge_mask = edges > 0
        density = float(np.mean(edge_mask))
        if density < _MIN_CANNY_DENSITY:
            # Still too sparse (near-solid or extremely subtle): use the
            # strongest gradient pixels so orientation is not all-zero.
            mag_thr = float(np.percentile(magnitude, 90.0))
            edge_mask = magnitude >= max(mag_thr, 1.0)
            logger.debug(
                "edge_descriptor: Canny density %.5f < %.3f; "
                "using mag-percentile mask (thr=%.2f, dens=%.5f)",
                density,
                _MIN_CANNY_DENSITY,
                mag_thr,
                float(np.mean(edge_mask)),
            )

        edge_angles = angle[edge_mask]
        edge_weights = magnitude[edge_mask]

        if edge_angles.size == 0:
            return np.zeros(
                cls.ORIENTATION_BINS,
                dtype=np.float32,
            )

        histogram, _ = np.histogram(
            edge_angles,
            bins=cls.ORIENTATION_BINS,
            range=(0, 360),
            weights=edge_weights,
        )

        histogram = histogram.astype(np.float32)

        histogram /= (
            np.linalg.norm(histogram)
            + _EPS
        )

        return histogram

    @staticmethod
    def similarity(
        query_hist: np.ndarray,
        candidate_hist: np.ndarray,
    ) -> float:
        """
        Cosine similarity between edge histograms.

        Returns
        -------
        float

        1.0 = identical (including both empty / unstructured)

        0.0 = unrelated (or one structured and one empty)
        """

        q_norm = float(np.linalg.norm(query_hist))
        c_norm = float(np.linalg.norm(candidate_hist))

        # Both unstructured (true solid / no gradients): equally flat → similar.
        # One empty and one not: structured vs plain → dissimilar.
        if q_norm < _EPS and c_norm < _EPS:
            return 1.0
        if q_norm < _EPS or c_norm < _EPS:
            return 0.0

        return float(
            np.dot(
                query_hist,
                candidate_hist,
            )
            / (q_norm * c_norm)
        )

    @staticmethod
    def serialize(
        histogram: np.ndarray,
    ) -> bytes:
        return histogram.astype(
            np.float32
        ).tobytes()

    @staticmethod
    def deserialize(
        blob: bytes,
    ) -> np.ndarray:
        return np.frombuffer(
            blob,
            dtype=np.float32,
        )
