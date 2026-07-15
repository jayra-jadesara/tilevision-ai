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


class EdgeDescriptor:
    """
    Edge orientation descriptor.
    """

    ORIENTATION_BINS = 36

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

        edges = cv2.Canny(
            gray,
            threshold1=80,
            threshold2=180,
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

        edge_mask = edges > 0

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
            + 1e-8
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

        1.0 = identical

        0.0 = unrelated
        """

        denom = (
            np.linalg.norm(query_hist)
            * np.linalg.norm(candidate_hist)
            + 1e-8
        )

        return float(
            np.dot(
                query_hist,
                candidate_hist,
            )
            / denom
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