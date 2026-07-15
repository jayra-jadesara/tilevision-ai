"""
Pattern descriptor for TileVision AI.

Detects visual surface patterns such as:
- fine dots
- speckles
- terrazzo particles
- small blobs
- irregular texture regions

The descriptor measures not only blob count and size,
but also particle consistency and spatial distribution.

TileVision AI v2
"""

from __future__ import annotations

import cv2
import numpy as np


class PatternDescriptor:

    # ---------------------------------------------------------
    # Feature vector layout
    # ---------------------------------------------------------
    #
    # [
    #     0: density,
    #     1: mean_size,
    #     2: size_std,
    #     3: count_normalized,
    #     4: coverage,
    #     5: size_consistency,
    #     6: spatial_uniformity,
    #     7: small_blob_ratio,
    # ]
    #
    # IMPORTANT:
    # Changing this from 5 -> 8 requires re-indexing existing images.

    FEATURE_SIZE = 8

    IMAGE_SIZE = 256

    @classmethod
    def extract(
        cls,
        image_bgr: np.ndarray,
    ) -> np.ndarray:
        """
        Extract pattern statistics.

        Returns
        -------
        np.ndarray
            8-dimensional normalized pattern feature vector.
        """

        # ---------------------------------------------------------
        # 1. Convert to grayscale
        # ---------------------------------------------------------

        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY,
        )

        # ---------------------------------------------------------
        # 2. Normalize image size
        # ---------------------------------------------------------

        gray = cv2.resize(
            gray,
            (
                cls.IMAGE_SIZE,
                cls.IMAGE_SIZE,
            ),
            interpolation=cv2.INTER_AREA,
        )

        # ---------------------------------------------------------
        # 3. Reduce very small image noise
        # ---------------------------------------------------------

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )

        # ---------------------------------------------------------
        # 4. Remove gradual background / lighting variation
        # ---------------------------------------------------------

        background = cv2.GaussianBlur(
            gray,
            (0, 0),
            sigmaX=7,
        )

        detail = cv2.absdiff(
            gray,
            background,
        )

        # ---------------------------------------------------------
        # 5. Automatic threshold
        # ---------------------------------------------------------

        _, binary = cv2.threshold(
            detail,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # ---------------------------------------------------------
        # 6. Remove isolated pixel noise
        # ---------------------------------------------------------

        kernel = np.ones(
            (2, 2),
            dtype=np.uint8,
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
        )

        # ---------------------------------------------------------
        # 7. Connected components
        # ---------------------------------------------------------

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8,
            )
        )

        height, width = gray.shape

        image_area = float(
            width * height
        )

        blob_areas = []
        blob_centroids = []

        # ---------------------------------------------------------
        # 8. Extract valid pattern blobs
        # ---------------------------------------------------------

        for i in range(
            1,
            num_labels,
        ):

            area = float(
                stats[
                    i,
                    cv2.CC_STAT_AREA,
                ]
            )

            # Ignore tiny noise.
            if area < 3:
                continue

            # Ignore very large regions.
            #
            # Large connected regions usually represent:
            # - marble veins
            # - shadows
            # - image boundaries
            # - large decorative shapes
            #
            # rather than fine particles.

            if area > image_area * 0.025:
                continue

            blob_areas.append(
                area
            )

            blob_centroids.append(
                centroids[i]
            )

        # ---------------------------------------------------------
        # 9. No detected pattern
        # ---------------------------------------------------------

        if not blob_areas:

            return np.zeros(
                cls.FEATURE_SIZE,
                dtype=np.float32,
            )

        areas = np.asarray(
            blob_areas,
            dtype=np.float32,
        )

        centroids_array = np.asarray(
            blob_centroids,
            dtype=np.float32,
        )

        count = len(areas)

        total_blob_area = float(
            np.sum(areas)
        )

        # ---------------------------------------------------------
        # Feature 1: Density
        # ---------------------------------------------------------

        density = (
            count
            / image_area
        )

        # ---------------------------------------------------------
        # Feature 2: Mean blob size
        # ---------------------------------------------------------

        mean_area = float(
            np.mean(areas)
        )

        mean_size = (
            mean_area
            / image_area
        )

        # ---------------------------------------------------------
        # Feature 3: Blob size standard deviation
        # ---------------------------------------------------------

        size_std = (
            float(
                np.std(areas)
            )
            / image_area
        )

        # ---------------------------------------------------------
        # Feature 4: Normalized blob count
        # ---------------------------------------------------------

        count_normalized = min(
            count / 500.0,
            1.0,
        )

        # ---------------------------------------------------------
        # Feature 5: Surface coverage
        # ---------------------------------------------------------

        coverage = min(
            total_blob_area
            / image_area,
            1.0,
        )

        # ---------------------------------------------------------
        # Feature 6: Size consistency
        # ---------------------------------------------------------
        #
        # Fine speckles generally have similar particle sizes.
        #
        # 1.0 = highly consistent particle sizes
        # 0.0 = highly irregular particle sizes

        raw_size_std = float(
            np.std(areas)
        )

        coefficient_of_variation = (
            raw_size_std
            / (
                mean_area
                + 1e-8
            )
        )

        size_consistency = (
            1.0
            / (
                1.0
                + coefficient_of_variation
            )
        )

        size_consistency = float(
            np.clip(
                size_consistency,
                0.0,
                1.0,
            )
        )

        # ---------------------------------------------------------
        # Feature 7: Spatial uniformity
        # ---------------------------------------------------------
        #
        # Split image into a 4x4 grid.
        #
        # Fine dotted/speckled tiles usually distribute particles
        # across most of the tile.
        #
        # Marble or localized decoration tends to concentrate
        # details in only certain regions.

        grid_size = 4

        grid_counts = np.zeros(
            (
                grid_size,
                grid_size,
            ),
            dtype=np.float32,
        )

        for centroid in centroids_array:

            x = float(
                centroid[0]
            )

            y = float(
                centroid[1]
            )

            grid_x = min(
                int(
                    x
                    / width
                    * grid_size
                ),
                grid_size - 1,
            )

            grid_y = min(
                int(
                    y
                    / height
                    * grid_size
                ),
                grid_size - 1,
            )

            grid_counts[
                grid_y,
                grid_x,
            ] += 1.0

        grid_mean = float(
            np.mean(
                grid_counts
            )
        )

        grid_std = float(
            np.std(
                grid_counts
            )
        )

        if grid_mean > 0:

            spatial_variation = (
                grid_std
                / (
                    grid_mean
                    + 1e-8
                )
            )

            spatial_uniformity = (
                1.0
                / (
                    1.0
                    + spatial_variation
                )
            )

        else:

            spatial_uniformity = 0.0

        spatial_uniformity = float(
            np.clip(
                spatial_uniformity,
                0.0,
                1.0,
            )
        )

        # ---------------------------------------------------------
        # Feature 8: Small blob ratio
        # ---------------------------------------------------------
        #
        # Percentage of detected components that are relatively
        # small.
        #
        # Fine dotted/speckled surfaces should normally contain
        # many small components.

        small_blob_threshold = 25.0

        small_blob_count = int(
            np.sum(
                areas
                <= small_blob_threshold
            )
        )

        small_blob_ratio = (
            small_blob_count
            / max(
                count,
                1,
            )
        )

        # ---------------------------------------------------------
        # Final feature vector
        # ---------------------------------------------------------

        return np.asarray(
            [
                density,
                mean_size,
                size_std,
                count_normalized,
                coverage,
                size_consistency,
                spatial_uniformity,
                small_blob_ratio,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def similarity(
        query: np.ndarray,
        candidate: np.ndarray,
    ) -> float:
        """
        Compare two pattern feature vectors.

        Uses weighted feature distance.

        More importance is given to:
        - spatial distribution
        - particle-size consistency
        - small particle ratio
        """

        query = np.asarray(
            query,
            dtype=np.float32,
        ).reshape(-1)

        candidate = np.asarray(
            candidate,
            dtype=np.float32,
        ).reshape(-1)

        if (
            query.size == 0
            or candidate.size == 0
        ):
            return 0.0

        if query.shape != candidate.shape:
            return 0.0

        # ---------------------------------------------------------
        # Backward compatibility
        # ---------------------------------------------------------

        if query.size == 5:

            distance = float(
                np.linalg.norm(
                    query
                    - candidate
                )
            )

            return float(
                1.0
                / (
                    1.0
                    + distance * 10.0
                )
            )

        if query.size != 8:
            return 0.0

        # ---------------------------------------------------------
        # Weighted feature comparison
        # ---------------------------------------------------------

        weights = np.asarray(
            [
                0.05,  # density
                0.05,  # mean size
                0.05,  # size std
                0.15,  # blob count
                0.15,  # coverage
                0.20,  # size consistency
                0.20,  # spatial uniformity
                0.15,  # small blob ratio
            ],
            dtype=np.float32,
        )

        difference = np.abs(
            query
            - candidate
        )

        weighted_distance = float(
            np.sum(
                difference
                * weights
            )
        )

        similarity = (
            1.0
            / (
                1.0
                + weighted_distance * 5.0
            )
        )

        return float(
            np.clip(
                similarity,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def serialize(
        features: np.ndarray,
    ) -> bytes:

        return np.asarray(
            features,
            dtype=np.float32,
        ).tobytes()

    @staticmethod
    def deserialize(
        blob: bytes,
    ) -> np.ndarray:

        return np.frombuffer(
            blob,
            dtype=np.float32,
        ).copy()