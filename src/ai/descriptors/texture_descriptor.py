"""
Texture descriptor for TileVision AI.

Uses Multi-Scale Local Binary Pattern (LBP) to describe tile
surface textures at different spatial scales.

Designed to improve discrimination between:

• Dotted / Speckled tiles
• Marble
• Granite
• Wood
• Cement
• Stone
• Plain tiles

Author:
TileVision AI v2
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


logger = logging.getLogger(
    "tilevision.ai.texture_descriptor"
)


class TextureDescriptor:
    """
    Multi-scale Local Binary Pattern texture descriptor.

    A single LBP scale often produces very similar histogram
    distributions for different tile patterns.

    Multi-scale LBP captures:

    - Fine surface details
    - Medium texture structures
    - Larger pattern structures
    """

    # ---------------------------------------------------------
    # Multi-scale LBP configuration
    # ---------------------------------------------------------

    SCALES = (
        # Fine texture:
        # dots, grains, speckles
        (1, 8),

        # Medium texture
        (2, 16),

        # Larger texture structures
        (3, 24),
    )

    # ---------------------------------------------------------

    @classmethod
    def extract(
        cls,
        image_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Extract normalized multi-scale LBP descriptor.

        Returns
        -------
        np.ndarray
            Concatenated normalized LBP histograms.
        """

        if image_bgr is None:
            raise ValueError(
                "TextureDescriptor received an empty image."
            )

        # Convert to grayscale
        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY,
        )

        # Reduce sensitivity to lighting differences
        gray = cv2.equalizeHist(gray)

        descriptors = []

        # -----------------------------------------------------
        # Extract LBP at multiple scales
        # -----------------------------------------------------

        for radius, points in cls.SCALES:

            lbp = local_binary_pattern(
                gray,
                P=points,
                R=radius,
                method="uniform",
            )

            bins = points + 2

            hist, _ = np.histogram(
                lbp.ravel(),
                bins=np.arange(
                    0,
                    bins + 1,
                ),
                range=(
                    0,
                    bins,
                ),
            )

            hist = hist.astype(
                np.float32
            )

            # Normalize each scale independently
            hist /= (
                hist.sum()
                + 1e-8
            )

            descriptors.append(
                hist
            )

        # -----------------------------------------------------
        # Combine all texture scales
        # -----------------------------------------------------

        descriptor = np.concatenate(
            descriptors
        ).astype(
            np.float32
        )

        # L1 normalization
        descriptor /= (
            descriptor.sum()
            + 1e-8
        )

        return descriptor

    # ---------------------------------------------------------

    @staticmethod
    def similarity(
        query_hist: np.ndarray,
        candidate_hist: np.ndarray,
    ) -> float:
        """
        Calculate texture similarity.

        Uses Bhattacharyya distance instead of histogram
        correlation.

        Returns
        -------
        float
            Similarity approximately between 0 and 1.

            1.0 = highly similar
            0.0 = highly different
        """

        query_hist = np.asarray(
            query_hist,
            dtype=np.float32,
        ).flatten()

        candidate_hist = np.asarray(
            candidate_hist,
            dtype=np.float32,
        ).flatten()

        # Different descriptor versions cannot be compared.
        if query_hist.shape != candidate_hist.shape:
            return 0.0

        distance = cv2.compareHist(
            query_hist,
            candidate_hist,
            cv2.HISTCMP_BHATTACHARYYA,
        )

        similarity = 1.0 - float(
            distance
        )

        return max(
            0.0,
            min(
                1.0,
                similarity,
            ),
        )

    # ---------------------------------------------------------

    @staticmethod
    def serialize(
        histogram: np.ndarray,
    ) -> bytes:
        """
        Serialize descriptor for SQLite storage.
        """

        return np.asarray(
            histogram,
            dtype=np.float32,
        ).tobytes()

    # ---------------------------------------------------------

    @staticmethod
    def deserialize(
        blob: bytes,
    ) -> np.ndarray:
        """
        Deserialize descriptor from SQLite storage.
        """

        return np.frombuffer(
            blob,
            dtype=np.float32,
        ).copy()