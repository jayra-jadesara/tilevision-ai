"""
DINOv2 embedder module for TileVision AI.

Uses Meta DINOv2 with a batched multi-scale strategy:

1. Full tile image   (global context)
2. Center crop       (large region, ~65%)
3. Detail crop       (fine pattern region, ~40%)

All views are embedded in a single forward pass, then fused with
fixed weights into a 1024D L2-normalized vector compatible with FAISS.

DINOv2 Large: 1024 dimensions
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModel

from src.ai.models import PreprocessedImage
from src.ai.inference_guard import synchronized_inference
from src.ai.gpu_info import DevicePreference, detect_gpu_runtime
from src.ai.preprocess.image_preprocessor import ImagePreprocessor

logger = logging.getLogger("tilevision.ai.embedder")

# Weighted fusion of multi-scale views.  Global dominates; detail
# boosts fine-grained pattern discrimination without overpowering semantics.
_VIEW_WEIGHTS: Tuple[float, ...] = (0.50, 0.30, 0.20)


class DINOv2Embedder:

    MODEL_NAME = "facebook/dinov2-large"
    EMBEDDING_DIM = 1024

    def __init__(self, *, device_preference: DevicePreference = "auto") -> None:
        self._device_preference: DevicePreference = device_preference
        self._runtime = detect_gpu_runtime(preference=device_preference)
        self._device = torch.device(self._runtime.active_device)
        self._processor = None
        self._model = None

        logger.info(self._runtime.summary_for_log())
        logger.info(
            "DINOv2 Embedder initialized. Device: %s",
            self._device.type.upper(),
        )

    @property
    def using_gpu(self) -> bool:
        return self._device.type == "cuda"

    @property
    def runtime_info(self):
        return self._runtime

    def load_model(self) -> None:
        if self._model is not None:
            return

        logger.info("Loading DINOv2 model...")

        self._processor = AutoImageProcessor.from_pretrained(self.MODEL_NAME)
        self._model = AutoModel.from_pretrained(self.MODEL_NAME)
        self._model.to(self._device)
        self._model.eval()

        if self._device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            logger.info(
                "CUDA GPU: %s (%.1f GB VRAM)",
                self._runtime.device_name,
                self._runtime.vram_gb or 0.0,
            )
        else:
            thread_count = min(8, os.cpu_count() or 4)
            torch.set_num_threads(thread_count)
            logger.info("CPU inference threads: %d", thread_count)

        logger.info("DINOv2 model loaded successfully.")

    def _forward_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Single DINOv2 forward pass."""
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {
            key: value.to(self._device, non_blocking=True)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            if self._device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    outputs = self._model(**inputs)
            else:
                outputs = self._model(**inputs)

        embeddings = (
            outputs.last_hidden_state[:, 0]
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        return embeddings / norms

    @staticmethod
    def _generate_views(image: Image.Image) -> List[Image.Image]:
        """
        Build global + center + detail views from a preprocessed PIL image.
        """
        image = image.convert("RGB")
        width, height = image.size
        views: List[Image.Image] = [image]

        if width < 64 or height < 64:
            return views

        center_w = max(1, int(width * 0.65))
        center_h = max(1, int(height * 0.65))
        center_left = (width - center_w) // 2
        center_top = (height - center_h) // 2
        views.append(
            image.crop(
                (
                    center_left,
                    center_top,
                    center_left + center_w,
                    center_top + center_h,
                )
            )
        )

        detail_w = max(1, int(width * 0.40))
        detail_h = max(1, int(height * 0.40))
        detail_left = (width - detail_w) // 2
        detail_top = (height - detail_h) // 2
        views.append(
            image.crop(
                (
                    detail_left,
                    detail_top,
                    detail_left + detail_w,
                    detail_top + detail_h,
                )
            )
        )

        return views

    def _extract_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Run DINOv2 on a list of PIL images in one batched forward pass.

        Returns:
            (N, 1024) array of L2-normalized per-view embeddings.
        """
        if self._model is None:
            self.load_model()

        with synchronized_inference():
            try:
                return self._forward_batch(images)
            except RuntimeError as exc:
                message = str(exc).lower()
                is_oom = "out of memory" in message or "cuda error" in message
                if not is_oom or self._device.type != "cuda" or len(images) <= 1:
                    raise

                logger.warning(
                    "CUDA OOM on batch of %d views — splitting and retrying.",
                    len(images),
                )
                torch.cuda.empty_cache()
                mid = len(images) // 2
                left = self._extract_batch(images[:mid])
                right = self._extract_batch(images[mid:])
                return np.vstack([left, right])

    @staticmethod
    def _fuse_embeddings(
        view_embeddings: np.ndarray,
        weights: Tuple[float, ...] = _VIEW_WEIGHTS,
    ) -> np.ndarray:
        """
        Weighted combination of per-view embeddings, then L2-normalize.
        """
        n_views = view_embeddings.shape[0]
        w = np.asarray(weights[:n_views], dtype=np.float32)
        w /= w.sum()

        fused = (view_embeddings * w[:, np.newaxis]).sum(axis=0).astype(np.float32)
        fused /= np.linalg.norm(fused) + 1e-8
        return fused

    def extract_from_preprocessed(
        self,
        processed: PreprocessedImage,
    ) -> np.ndarray:
        """
        Extract a multi-scale DINOv2 embedding from an already-preprocessed image.

        This is the primary entry point — avoids reloading/resizing the image.
        """
        views = self._generate_views(processed.pil)
        view_embeddings = self._extract_batch(views)
        final_embedding = self._fuse_embeddings(view_embeddings)

        logger.debug(
            "Multi-scale DINOv2 embedding: views=%d dimension=%d",
            len(views),
            final_embedding.shape[0],
        )
        return final_embedding

    def extract_batch_from_preprocessed(
        self,
        processed_images: List[PreprocessedImage],
    ) -> List[np.ndarray]:
        """
        Extract embeddings for multiple preprocessed images.

        All views across the batch are run in a single DINOv2 forward pass
        for better throughput during folder indexing.
        """
        if not processed_images:
            return []

        all_views: List[Image.Image] = []
        view_counts: List[int] = []

        for processed in processed_images:
            views = self._generate_views(processed.pil)
            view_counts.append(len(views))
            all_views.extend(views)

        view_embeddings = self._extract_batch(all_views)

        results: List[np.ndarray] = []
        offset = 0
        for count in view_counts:
            chunk = view_embeddings[offset : offset + count]
            results.append(self._fuse_embeddings(chunk))
            offset += count

        logger.debug(
            "Batched DINOv2 embeddings: images=%d views=%d",
            len(processed_images),
            len(all_views),
        )
        return results

    def extract(self, image_path: str) -> np.ndarray:
        """
        Extract embedding from a file path (loads + preprocesses once).

        Prefer extract_from_preprocessed() when the caller already has
        a PreprocessedImage to avoid duplicate I/O.
        """
        processed = ImagePreprocessor.preprocess(image_path)
        return self.extract_from_preprocessed(processed)

    def get_embedding(self, image_path: str) -> np.ndarray:
        """Backward-compatible alias for extract()."""
        return self.extract(image_path)
