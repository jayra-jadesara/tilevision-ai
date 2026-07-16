"""
Pattern descriptor for TileVision AI.

Detects visual surface patterns such as:
- fine dots / speckles (round, isotropic particles)
- marble veins (elongated, directional structures)
- terrazzo particles
- plain surfaces

v3 adds structure features (elongation, circularity, vein coverage,
structure coherence) to improve speckled vs marble discrimination.
"""

from __future__ import annotations

import cv2
import numpy as np


class PatternDescriptor:

    # ---------------------------------------------------------
    # Feature vector layout (v3 — 12 dimensions)
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
    #     8: elongation_ratio,      # fraction of blobs with aspect > 2.5
    #     9: mean_circularity,       # round particles score higher
    #    10: vein_coverage,          # area fraction from elongated blobs
    #    11: structure_coherence,    # directional veining (structure tensor)
    # ]
    #
    # Bump pattern_feature_version when this layout changes.

    FEATURE_SIZE = 12
    LEGACY_FEATURE_SIZE = 8

    IMAGE_SIZE = 256
    _ELONGATION_THRESHOLD = 2.5
    _VEIN_ASPECT_THRESHOLD = 2.0
    _SMALL_BLOB_AREA = 25.0
    _MAX_BLOB_AREA_RATIO = 0.025

    @classmethod
    def _prepare_detail_image(
        cls,
        image_bgr: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(
            gray,
            (cls.IMAGE_SIZE, cls.IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=7)
        detail = cv2.absdiff(gray, background)

        _, binary = cv2.threshold(
            detail,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        kernel = np.ones((2, 2), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        image_area = float(cls.IMAGE_SIZE * cls.IMAGE_SIZE)
        return detail, image_area, binary

    @staticmethod
    def _structure_coherence(detail: np.ndarray) -> float:
        """Mean structure-tensor coherence on significant gradient pixels."""
        gx = cv2.Sobel(detail, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(detail, cv2.CV_32F, 0, 1, ksize=3)

        j11 = cv2.boxFilter(gx * gx, ddepth=-1, ksize=(5, 5))
        j12 = cv2.boxFilter(gx * gy, ddepth=-1, ksize=(5, 5))
        j22 = cv2.boxFilter(gy * gy, ddepth=-1, ksize=(5, 5))

        trace = j11 + j22
        det = j11 * j22 - j12 * j12
        discriminant = np.maximum(trace * trace - 4.0 * det, 0.0)
        sqrt_disc = np.sqrt(discriminant)

        lambda1 = 0.5 * (trace + sqrt_disc)
        lambda2 = 0.5 * (trace - sqrt_disc)
        coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-8)

        magnitude = np.sqrt(gx * gx + gy * gy)
        threshold = float(np.percentile(magnitude, 60))
        mask = magnitude > threshold
        if not np.any(mask):
            return 0.0

        return float(np.clip(np.mean(coherence[mask]), 0.0, 1.0))

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
            12-dimensional normalized pattern feature vector.
        """
        detail, image_area, binary = cls._prepare_detail_image(image_bgr)

        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        height, width = cls.IMAGE_SIZE, cls.IMAGE_SIZE
        max_blob_area = image_area * cls._MAX_BLOB_AREA_RATIO

        blob_areas: list[float] = []
        blob_centroids: list[np.ndarray] = []
        blob_elongations: list[float] = []
        blob_circularities: list[float] = []

        for i in range(1, num_labels):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < 3 or area > max_blob_area:
                continue

            blob_w = float(stats[i, cv2.CC_STAT_WIDTH])
            blob_h = float(stats[i, cv2.CC_STAT_HEIGHT])
            short_side = max(min(blob_w, blob_h), 1.0)
            long_side = max(blob_w, blob_h)
            elongation = long_side / short_side
            circularity = short_side / long_side

            blob_areas.append(area)
            blob_centroids.append(centroids[i])
            blob_elongations.append(elongation)
            blob_circularities.append(circularity)

        if not blob_areas:
            return np.zeros(cls.FEATURE_SIZE, dtype=np.float32)

        areas = np.asarray(blob_areas, dtype=np.float32)
        centroids_array = np.asarray(blob_centroids, dtype=np.float32)
        elongations = np.asarray(blob_elongations, dtype=np.float32)
        circularities = np.asarray(blob_circularities, dtype=np.float32)

        count = len(areas)
        total_blob_area = float(np.sum(areas))

        density = count / image_area
        mean_area = float(np.mean(areas))
        mean_size = mean_area / image_area
        size_std = float(np.std(areas)) / image_area
        count_normalized = min(count / 500.0, 1.0)
        coverage = min(total_blob_area / image_area, 1.0)

        raw_size_std = float(np.std(areas))
        coefficient_of_variation = raw_size_std / (mean_area + 1e-8)
        size_consistency = float(
            np.clip(1.0 / (1.0 + coefficient_of_variation), 0.0, 1.0)
        )

        grid_size = 4
        grid_counts = np.zeros((grid_size, grid_size), dtype=np.float32)
        for centroid in centroids_array:
            grid_x = min(int(centroid[0] / width * grid_size), grid_size - 1)
            grid_y = min(int(centroid[1] / height * grid_size), grid_size - 1)
            grid_counts[grid_y, grid_x] += 1.0

        grid_mean = float(np.mean(grid_counts))
        grid_std = float(np.std(grid_counts))
        if grid_mean > 0:
            spatial_variation = grid_std / (grid_mean + 1e-8)
            spatial_uniformity = float(
                np.clip(1.0 / (1.0 + spatial_variation), 0.0, 1.0)
            )
        else:
            spatial_uniformity = 0.0

        small_blob_count = int(np.sum(areas <= cls._SMALL_BLOB_AREA))
        small_blob_ratio = small_blob_count / max(count, 1)

        elongation_ratio = float(
            np.mean(elongations > cls._ELONGATION_THRESHOLD)
        )
        mean_circularity = float(np.clip(np.mean(circularities), 0.0, 1.0))

        vein_mask = elongations >= cls._VEIN_ASPECT_THRESHOLD
        if np.any(vein_mask):
            vein_coverage = float(np.sum(areas[vein_mask]) / (total_blob_area + 1e-8))
        else:
            vein_coverage = 0.0
        vein_coverage = float(np.clip(vein_coverage, 0.0, 1.0))

        structure_coherence = cls._structure_coherence(detail)

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
                elongation_ratio,
                mean_circularity,
                vein_coverage,
                structure_coherence,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def similarity(
        query: np.ndarray,
        candidate: np.ndarray,
    ) -> float:
        """
        Compare two pattern feature vectors using weighted feature distance.
        """
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)

        if query.size == 0 or candidate.size == 0:
            return 0.0
        if query.shape != candidate.shape:
            return 0.0

        if query.size == 5:
            distance = float(np.linalg.norm(query - candidate))
            return float(1.0 / (1.0 + distance * 10.0))

        if query.size == PatternDescriptor.LEGACY_FEATURE_SIZE:
            weights = np.asarray(
                [
                    0.05, 0.05, 0.05, 0.15, 0.15,
                    0.20, 0.20, 0.15,
                ],
                dtype=np.float32,
            )
            scale = 5.0
        elif query.size == PatternDescriptor.FEATURE_SIZE:
            weights = np.asarray(
                [
                    0.04,  # density
                    0.04,  # mean size
                    0.04,  # size std
                    0.10,  # blob count
                    0.10,  # coverage
                    0.12,  # size consistency
                    0.12,  # spatial uniformity
                    0.10,  # small blob ratio
                    0.12,  # elongation ratio
                    0.10,  # mean circularity
                    0.12,  # vein coverage
                    0.10,  # structure coherence
                ],
                dtype=np.float32,
            )
            scale = 5.0
        else:
            return 0.0

        difference = np.abs(query - candidate)
        weighted_distance = float(np.sum(difference * weights))
        similarity = 1.0 / (1.0 + weighted_distance * scale)

        return float(np.clip(similarity, 0.0, 1.0))

    @staticmethod
    def serialize(
        features: np.ndarray,
    ) -> bytes:
        return np.asarray(features, dtype=np.float32).tobytes()

    @staticmethod
    def deserialize(
        blob: bytes,
    ) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32).copy()
