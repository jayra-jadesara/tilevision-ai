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

import cv2
import numpy as np

from src.ai.embedder import DINOv2Embedder
from src.ai.models import TileFeatures
from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.descriptors.pattern_descriptor import PatternDescriptor

from src.ai.descriptors.color_descriptor import ColorDescriptor
from src.ai.descriptors.texture_descriptor import TextureDescriptor
from src.ai.descriptors.edge_descriptor import EdgeDescriptor
from src.ai.descriptors.pattern_descriptor import PatternDescriptor

logger = logging.getLogger("tilevision.ai.feature_extractor")


class FeatureExtractor:

    def __init__(
        self,
        embedder: DINOv2Embedder | None = None,
    ):
        self._embedder = embedder or DINOv2Embedder()

    # --------------------------------------------------------
    
    def load_model(self) -> None:
        self._embedder.load_model()

    @staticmethod
    def dominant_color(image_bgr):

        pixels = image_bgr.reshape((-1, 3))

        pixels = np.float32(pixels)

        criteria = (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            20,
            1.0,
        )

        k = 1

        _, labels, centers = cv2.kmeans(
            pixels,
            k,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS,
        )

        color = centers[0].astype(np.uint8)

        return (
            int(color[2]),
            int(color[1]),
            int(color[0]),
        )

    # --------------------------------------------------------

    def extract(
        self,
        image_path: str,
    ) -> TileFeatures:

        logger.info(
            "Extracting AI features: %s",
            image_path,
        )

        image = ImagePreprocessor.preprocess(
            image_path,
        )

        embedding = np.asarray(
            self._embedder.extract(
                image_path,
            ),
            dtype=np.float32,
        )

        color_hist = ColorDescriptor.extract(
            image.bgr,
        )

        texture_hist = TextureDescriptor.extract(
            image.bgr,
        )

        edge_hist = EdgeDescriptor.extract(
            image.bgr,
        )

        # Detect dots, speckles and small surface patterns
        pattern_features = PatternDescriptor.extract(
            image.bgr,
        )

        dominant = self.dominant_color(
            image.bgr,
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