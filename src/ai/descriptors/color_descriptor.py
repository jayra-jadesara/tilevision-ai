"""
Color descriptor for TileVision AI.

Extracts a normalized HSV histogram from an image.

Why HSV?
--------
HSV separates color from brightness much better than RGB.

Hue        -> Tile color
Saturation -> Richness
Value      -> Brightness

This makes similarity much more robust for ceramic tiles.

Author:
TileVision AI v2
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger("tilevision.ai.color_descriptor")


class ColorDescriptor:
    """
    HSV Histogram descriptor.

    Produces a normalized feature vector suitable for:

        • Similarity Search
        • Re-ranking
        • Duplicate Detection
    """

    # Histogram bins
    H_BINS = 32
    S_BINS = 32
    V_BINS = 8

    @classmethod
    def extract(
        cls,
        image_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Extract normalized HSV histogram.

        Parameters
        ----------
        image_bgr
            OpenCV BGR image

        Returns
        -------
        ndarray
            float32 normalized histogram
        """

        hsv = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2HSV,
        )

        histogram = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [
                cls.H_BINS,
                cls.S_BINS,
                cls.V_BINS,
            ],
            [
                0,
                180,
                0,
                256,
                0,
                256,
            ],
        )

        histogram = histogram.astype(np.float32)

        cv2.normalize(
            histogram,
            histogram,
            alpha=1.0,
            beta=0.0,
            norm_type=cv2.NORM_L2,
        )

        return histogram.flatten()

    @staticmethod
    def similarity(
        query_hist: np.ndarray,
        candidate_hist: np.ndarray,
    ) -> float:
        """
        Histogram correlation.

        Returns
        -------
        float

        1.0  -> identical colors

        0.0  -> unrelated

        -1.0 -> opposite
        """

        score = cv2.compareHist(
            query_hist.astype(np.float32),
            candidate_hist.astype(np.float32),
            cv2.HISTCMP_CORREL,
        )

        return float(score)

    @staticmethod
    def serialize(
        histogram: np.ndarray,
    ) -> bytes:
        """
        Convert histogram into bytes for SQLite.
        """

        return histogram.astype(
            np.float32
        ).tobytes()

    @staticmethod
    def deserialize(
        blob: bytes,
    ) -> np.ndarray:
        """
        Restore histogram from SQLite.
        """

        return np.frombuffer(
            blob,
            dtype=np.float32,
        )