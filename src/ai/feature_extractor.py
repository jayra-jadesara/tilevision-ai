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
from pathlib import Path
from typing import List, TYPE_CHECKING

import cv2
import numpy as np

from src.ai.embedder import DINOv2Embedder
from src.ai.models import TileFeatures, PreprocessedImage
from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.descriptors.color_descriptor import ColorDescriptor
from src.ai.descriptors.texture_descriptor import TextureDescriptor
from src.ai.descriptors.edge_descriptor import EdgeDescriptor
from src.ai.descriptors.pattern_descriptor import PatternDescriptor

if TYPE_CHECKING:
    from PIL import Image

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
        *,
        preprocess_workers: int = 4,
    ):
        self._embedder = embedder or DINOv2Embedder()
        self._preprocess_workers = max(1, int(preprocess_workers))
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
        *,
        for_query: bool = False,
    ) -> TileFeatures:
        """Extract full features when the image is already preprocessed."""
        total_start = time.perf_counter()

        t1 = time.perf_counter()
        embedding = np.asarray(
            self._embedder.extract_from_preprocessed(image, for_query=for_query),
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
        *,
        preprocess_workers: int | None = None,
    ) -> List[TileFeatures]:
        """Extract features for multiple image paths."""
        if not image_paths:
            return []

        if len(image_paths) == 1:
            return [self.extract(image_paths[0])]

        total_start = time.perf_counter()

        t0 = time.perf_counter()
        workers = preprocess_workers or self._preprocess_workers
        worker_count = min(max(1, workers), len(image_paths))
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
        *,
        for_query: bool = False,
    ) -> TileFeatures:

        logger.debug(
            "Extracting AI features: %s (for_query=%s)",
            image_path,
            for_query,
        )

        total_start = time.perf_counter()

        t0 = time.perf_counter()
        views: List[PreprocessedImage] = []
        if for_query:
            views = ImagePreprocessor.prepare_query_views(image_path, max_views=3)
            image = views[0]
        else:
            image = ImagePreprocessor.preprocess(image_path)
        preprocess_elapsed = time.perf_counter() - t0

        if for_query and len(views) > 1:
            features = self._extract_multi_view_query(views)
        else:
            features = self.extract_from_preprocessed(image, for_query=for_query)
        features_elapsed = time.perf_counter() - total_start

        self._last_timings = ExtractTimings(
            preprocessing=preprocess_elapsed,
            dinov2=self._last_timings.dinov2,
            descriptors=self._last_timings.descriptors,
            total=features_elapsed,
        )

        logger.debug(
            "Feature extract timing: preprocessing=%.3fs dinov2=%.3fs "
            "descriptors=%.3fs total=%.3fs views=%d",
            preprocess_elapsed,
            self._last_timings.dinov2,
            self._last_timings.descriptors,
            self._last_timings.total,
            max(1, len(views)),
        )

        return features

    def extract_index_vectors(
        self,
        image_path: str,
    ) -> tuple[TileFeatures, list[np.ndarray]]:
        """
        Index-time extract: primary TileFeatures plus optional aux FAISS vectors.

        Wide catalog sheets get a secondary texture-panel embedding (same tile
        id in FAISS) so a customer crop of the slab still retrieves the sheet.
        """
        features = self.extract(image_path, for_query=False)
        aux: list[np.ndarray] = []

        try:
            # Detect the panel on the raw sheet. Do NOT trim/content-crop the
            # full sheet first — that shifts the left/right split into the
            # text column and destroys texture-crop recall (measured).
            raw = ImagePreprocessor.load(image_path)
            raw = ImagePreprocessor.to_rgb(raw)
            panel = ImagePreprocessor.primary_texture_panel(raw)
            if panel is None:
                return features, aux

            panel = ImagePreprocessor.normalize_lighting(panel)
            panel = ImagePreprocessor.resize_letterbox(panel)
            rgb = ImagePreprocessor.to_numpy(panel)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            panel_image = PreprocessedImage(
                pil=panel,
                rgb=rgb,
                bgr=bgr,
                gray=gray,
                width=raw.size[0],
                height=raw.size[1],
            )
            panel_emb = np.asarray(
                self._embedder.extract_from_preprocessed(panel_image, for_query=False),
                dtype=np.float32,
            )
            # Skip near-duplicate aux vectors (ordinary tiles that slipped through).
            primary = np.asarray(features.embedding, dtype=np.float32).ravel()
            panel_v = panel_emb.ravel()
            sim = float(
                np.dot(primary, panel_v)
                / (np.linalg.norm(primary) * np.linalg.norm(panel_v) + 1e-8)
            )
            if sim < 0.97:
                aux.append(panel_emb)
                logger.info(
                    "Index aux texture-panel vector for %s (cos_vs_primary=%.3f)",
                    Path(image_path).name,
                    sim,
                )
        except Exception as exc:
            logger.warning(
                "Texture-panel aux embed skipped for %s: %s",
                image_path,
                exc,
            )

        return features, aux

    def extract_for_search(
        self,
        image_path: str,
        *,
        preloaded: Image.Image | None = None,
    ) -> tuple[TileFeatures, list[np.ndarray]]:
        """
        Query-only: one preprocess + one DINOv2 vector (reliable on Mac CPU).

        Multi-crop OpenCV recall is available via Auto Crop / Precise Crop.
        Drop-search must stay single-pass so results always return.

        Pass ``preloaded`` so the search use-case can decode the query once
        and reuse it for dHash + embedding.
        """
        total_start = time.perf_counter()
        t0 = time.perf_counter()
        # Always one view — never stack multi-crop DINOv2 on the drop path.
        views = ImagePreprocessor.prepare_query_views(
            image_path,
            max_views=1,
            preloaded=preloaded,
        )
        preprocess_elapsed = time.perf_counter() - t0

        embeddings: list[np.ndarray] = []
        dinov2_elapsed = 0.0
        for view in views:
            t1 = time.perf_counter()
            emb = np.asarray(
                self._embedder.extract_from_preprocessed(view, for_query=True),
                dtype=np.float32,
            )
            dinov2_elapsed += time.perf_counter() - t1
            embeddings.append(emb)

        features = self._fuse_query_embeddings(views[0], embeddings, dinov2_elapsed)
        self._last_timings = ExtractTimings(
            preprocessing=preprocess_elapsed,
            dinov2=self._last_timings.dinov2,
            descriptors=self._last_timings.descriptors,
            total=time.perf_counter() - total_start,
        )
        logger.info(
            "Search extract (single-pass): preprocess=%.2fs dinov2=%.2fs total=%.2fs",
            preprocess_elapsed,
            self._last_timings.dinov2,
            self._last_timings.total,
        )
        return features, embeddings

    def _extract_multi_view_query(
        self,
        views: List[PreprocessedImage],
    ) -> TileFeatures:
        """
        Embed several query crops and fuse DINOv2 vectors (L2-normalized mean).

        Descriptors come from the primary (best) crop. Query-only — does not
        change indexed catalog vectors.
        """
        embeddings: list[np.ndarray] = []
        dinov2_elapsed = 0.0

        for view in views:
            t1 = time.perf_counter()
            emb = np.asarray(
                self._embedder.extract_from_preprocessed(view, for_query=True),
                dtype=np.float32,
            )
            dinov2_elapsed += time.perf_counter() - t1
            embeddings.append(emb)

        return self._fuse_query_embeddings(views[0], embeddings, dinov2_elapsed)

    def _fuse_query_embeddings(
        self,
        primary: PreprocessedImage,
        embeddings: list[np.ndarray],
        dinov2_elapsed: float,
    ) -> TileFeatures:
        total_start = time.perf_counter()
        stacked = np.vstack(embeddings)
        fused = stacked.mean(axis=0)
        fused = fused / (np.linalg.norm(fused) + 1e-8)
        fused = fused.astype(np.float32)

        t2 = time.perf_counter()
        (
            color_hist,
            texture_hist,
            edge_hist,
            pattern_features,
            dominant,
        ) = self.extract_descriptors_from_preprocessed(primary)
        descriptors_elapsed = time.perf_counter() - t2

        self._last_timings = ExtractTimings(
            preprocessing=0.0,
            dinov2=dinov2_elapsed,
            descriptors=descriptors_elapsed,
            total=time.perf_counter() - total_start + dinov2_elapsed,
        )

        logger.info(
            "Query multi-crop DINOv2 fuse: views=%d dim=%d",
            len(embeddings),
            fused.shape[0],
        )

        return TileFeatures(
            embedding=fused,
            color_histogram=color_hist,
            texture_histogram=texture_hist,
            edge_histogram=edge_hist,
            pattern_features=pattern_features,
            dominant_color=dominant,
            width=primary.width,
            height=primary.height,
        )
