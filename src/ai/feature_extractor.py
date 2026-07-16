"""
Central AI feature extraction service.

This class is the ONLY place that knows how AI features are generated.

Pipeline

Image
   │
   ▼
ImagePreprocessor
   │
   ▼
DINOv2
   │
   ▼
HSV
   │
   ▼
LBP
   │
   ▼
Edge
   │
   ▼
Dominant Color
   │
   ▼
TileFeatures

Author:
TileVision AI v2
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

import numpy as np

from src.ai.embedder import DINOv2Embedder
from src.ai.models import TileFeatures
from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.descriptors.color_descriptor import ColorDescriptor
from src.ai.descriptors.texture_descriptor import TextureDescriptor
from src.ai.descriptors.edge_descriptor import EdgeDescriptor
from src.ai.descriptors.pattern_descriptor import PatternDescriptor

logger = logging.getLogger("tilevision.ai.feature_extractor")


@dataclass(slots=True)
class ExtractTimings:
    preprocessing: float = 0.0
    dinov2: float = 0.0
    descriptors: float = 0.0
    total: float = 0.0


class FeatureExtractor:

    def __init__(
        self,
        embedder: DINOv2Embedder | None = None,
    ):
        self._embedder = embedder or DINOv2Embedder()
        self._last_timings = ExtractTimings()

    @property
    def last_timings(self) -> ExtractTimings:
        return self._last_timings

    # --------------------------------------------------------
    
    def load_model(self) -> None:
        self._embedder.load_model()

    @staticmethod
    def dominant_color(image_bgr):
        return ColorDescriptor.dominant_color_rgb(image_bgr)

    # --------------------------------------------------------

    def extract_descriptors_from_preprocessed(
        self,
        image: PreprocessedImage,
    ) -> tuple:
        """Return handcrafted descriptors from a preprocessed image."""
        color_hist = ColorDescriptor.extract(image.bgr)
        texture_hist = TextureDescriptor.extract(image.bgr)
        edge_hist = EdgeDescriptor.extract(image.bgr)
        pattern_features = PatternDescriptor.extract(image.bgr)
        dominant = self.dominant_color(image.bgr)
        return color_hist, texture_hist, edge_hist, pattern_features, dominant

    def extract_from_preprocessed(
        self,
        image: PreprocessedImage,
    ) -> TileFeatures:
        """Extract full features when the image is already preprocessed."""
        total_start = time.perf_counter()

        t1 = time.perf_counter()
        embedding = np.asarray(
            self._embedder.extract_from_preprocessed(image),
            dtype=np.float32,
        )
        dinov2_elapsed = time.perf_counter() - t1

        t2 = time.perf_counter()
        (
            color_hist,
            texture_hist,
            edge_hist,
            pattern_features,
            dominant,
        ) = self.extract_descriptors_from_preprocessed(image)
        descriptors_elapsed = time.perf_counter() - t2

        self._last_timings = ExtractTimings(
            preprocessing=0.0,
            dinov2=dinov2_elapsed,
            descriptors=descriptors_elapsed,
            total=time.perf_counter() - total_start,
        )

        return TileFeatures(
            embedding=embedding,
            color_histogram=color_hist,
            texture_histogram=texture_hist,
            edge_histogram=edge_hist,
            pattern_features=pattern_features,
            dominant_color=dominant,
            width=image.width,
            height=image.height,
        )

    def extract_batch(
        self,
        image_paths: List[str],
    ) -> List[TileFeatures]:
        """Extract features for multiple image paths."""
        if not image_paths:
            return []

        if len(image_paths) == 1:
            return [self.extract(image_paths[0])]

        total_start = time.perf_counter()

        t0 = time.perf_counter()
        worker_count = min(4, len(image_paths))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            processed_images = list(pool.map(ImagePreprocessor.preprocess, image_paths))
        preprocess_elapsed = time.perf_counter() - t0

        t1 = time.perf_counter()
        embeddings = self._embedder.extract_batch_from_preprocessed(
            processed_images
        )
        dinov2_elapsed = time.perf_counter() - t1

        t2 = time.perf_counter()
        features_list: List[TileFeatures] = []

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            descriptor_results = list(
                pool.map(self.extract_descriptors_from_preprocessed, processed_images)
            )

        for processed, embedding, descriptor_tuple in zip(
            processed_images,
            embeddings,
            descriptor_results,
        ):
            (
                color_hist,
                texture_hist,
                edge_hist,
                pattern_features,
                dominant,
            ) = descriptor_tuple

            features_list.append(
                TileFeatures(
                    embedding=np.asarray(embedding, dtype=np.float32),
                    color_histogram=color_hist,
                    texture_histogram=texture_hist,
                    edge_histogram=edge_hist,
                    pattern_features=pattern_features,
                    dominant_color=dominant,
                    width=processed.width,
                    height=processed.height,
                )
            )

        descriptors_elapsed = time.perf_counter() - t2
        batch_size = len(image_paths)

        self._last_timings = ExtractTimings(
            preprocessing=preprocess_elapsed / batch_size,
            dinov2=dinov2_elapsed / batch_size,
            descriptors=descriptors_elapsed / batch_size,
            total=(time.perf_counter() - total_start) / batch_size,
        )

        logger.debug(
            "Batch feature extract: count=%d preprocessing=%.3fs dinov2=%.3fs "
            "descriptors=%.3fs",
            batch_size,
            preprocess_elapsed,
            dinov2_elapsed,
            descriptors_elapsed,
        )

        return features_list

    def extract(
        self,
        image_path: str,
    ) -> TileFeatures:

        logger.debug(
            "Extracting AI features: %s",
            image_path,
        )

        total_start = time.perf_counter()

        t0 = time.perf_counter()
        image = ImagePreprocessor.preprocess(image_path)
        preprocess_elapsed = time.perf_counter() - t0

        features = self.extract_from_preprocessed(image)
        features_elapsed = time.perf_counter() - total_start

        self._last_timings = ExtractTimings(
            preprocessing=preprocess_elapsed,
            dinov2=self._last_timings.dinov2,
            descriptors=self._last_timings.descriptors,
            total=features_elapsed,
        )

        logger.debug(
            "Feature extract timing: preprocessing=%.3fs dinov2=%.3fs "
            "descriptors=%.3fs total=%.3fs",
            preprocess_elapsed,
            self._last_timings.dinov2,
            self._last_timings.descriptors,
            self._last_timings.total,
        )

        return features