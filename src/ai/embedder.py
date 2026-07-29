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
from src.ai.inference_guard import (
    DEFAULT_INDEX_LOCK_TIMEOUT_S,
    DEFAULT_SEARCH_LOCK_TIMEOUT_S,
    synchronized_inference,
)
from src.ai.gpu_info import (
    DevicePreference,
    configure_mps_fallback,
    detect_gpu_runtime,
    is_mps_unsupported_op_error,
    mps_autocast_supported,
)
from src.ai.preprocess.image_preprocessor import ImagePreprocessor

logger = logging.getLogger("tilevision.ai.embedder")

# Weighted fusion of multi-scale views.  Global dominates; detail
# boosts fine-grained pattern discrimination without overpowering semantics.
_VIEW_WEIGHTS: Tuple[float, ...] = (0.50, 0.30, 0.20)


def _is_device_oom_error(device_type: str, message: str) -> bool:
    """True only for genuine out-of-memory failures (not missing MPS ops)."""
    text = (message or "").lower()
    if "out of memory" in text or "insufficient memory" in text:
        return True
    # Do NOT treat bare "mps" as OOM — that matched "not implemented for MPS"
    # and hid the real Mac search crash behind a useless batch-split retry.
    if device_type == "cuda" and "cuda error" in text and "memory" in text:
        return True
    return False


class DINOv2Embedder:

    MODEL_NAME = "facebook/dinov2-large"
    EMBEDDING_DIM = 1024

    def __init__(self, *, device_preference: DevicePreference = "auto") -> None:
        configure_mps_fallback()
        self._device_preference: DevicePreference = device_preference
        self._runtime = detect_gpu_runtime(preference=device_preference)
        self._device = torch.device(self._runtime.active_device)
        self._processor = None
        self._model = None
        self._mps_cpu_fallback_done = False

        logger.info(self._runtime.summary_for_log())
        logger.info(
            "DINOv2 Embedder initialized. Device: %s",
            self._device.type.upper(),
        )

    @property
    def using_gpu(self) -> bool:
        return self._device.type in ("cuda", "mps")

    @property
    def runtime_info(self):
        return self._runtime

    def load_model(self) -> None:
        if self._model is not None:
            return

        logger.info("Loading DINOv2 model...")

        from src.ai.model_paths import resolve_dinov2_model_source

        model_source, local_only = resolve_dinov2_model_source()
        logger.info(
            "DINOv2 source: %s (%s)",
            model_source,
            "offline/local" if local_only else "Hugging Face hub",
        )

        self._processor = AutoImageProcessor.from_pretrained(
            model_source,
            local_files_only=local_only,
        )
        self._model = AutoModel.from_pretrained(
            model_source,
            local_files_only=local_only,
        )
        self._model.to(self._device)
        self._model.eval()

        if self._device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            logger.info(
                "CUDA GPU: %s (%.1f GB VRAM)",
                self._runtime.device_name,
                self._runtime.vram_gb or 0.0,
            )
        elif self._device.type == "mps":
            configure_mps_fallback()
            logger.info("Apple GPU (MPS): %s", self._runtime.device_name)
        else:
            thread_count = min(8, os.cpu_count() or 4)
            torch.set_num_threads(thread_count)
            logger.info("CPU inference threads: %d", thread_count)

        logger.info("DINOv2 model loaded successfully.")

    def _fallback_mps_to_cpu(self, reason: str) -> None:
        """Move the model to CPU after an unimplemented MPS operator."""
        if self._mps_cpu_fallback_done and self._device.type == "cpu":
            return
        short = reason.splitlines()[0][:160]
        logger.warning(
            "MPS operator unavailable — switching DINOv2 to CPU so search continues. (%s)",
            short,
        )
        self._device = torch.device("cpu")
        self._device_preference = "cpu"
        self._runtime = detect_gpu_runtime(preference="cpu")
        if self._model is not None:
            self._model.to(self._device)
            self._model.eval()
        thread_count = min(8, os.cpu_count() or 4)
        torch.set_num_threads(thread_count)
        self._mps_cpu_fallback_done = True

    def _run_model_forward(self, inputs: dict) -> object:
        """Run DINOv2 forward pass with autocast only when the device supports it."""
        if self._device.type == "cuda":
            with torch.autocast(device_type="cuda"):
                return self._model(**inputs)
        if self._device.type == "mps":
            if mps_autocast_supported():
                with torch.autocast(device_type="mps"):
                    return self._model(**inputs)
            logger.debug("MPS autocast unavailable — running float32 inference on MPS")
            return self._model(**inputs)
        return self._model(**inputs)

    def _forward_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Single DINOv2 forward pass."""
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {
            key: value.to(self._device, non_blocking=True)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self._run_model_forward(inputs)

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

    def _extract_batch(
        self,
        images: List[Image.Image],
        *,
        for_query: bool = False,
    ) -> np.ndarray:
        """
        Run DINOv2 on a list of PIL images in one batched forward pass.

        Returns:
            (N, 1024) array of L2-normalized per-view embeddings.
        """
        if self._model is None:
            self.load_model()

        # Search must not wait hours behind indexing / a stuck MPS forward.
        lock_timeout = (
            DEFAULT_SEARCH_LOCK_TIMEOUT_S if for_query else DEFAULT_INDEX_LOCK_TIMEOUT_S
        )

        # Query embeds on Apple Silicon: use CPU to avoid silent MPS hangs.
        if for_query and self._device.type == "mps" and not self._mps_cpu_fallback_done:
            self._fallback_mps_to_cpu("query search prefers CPU (avoid MPS hang)")

        with synchronized_inference(timeout=lock_timeout, purpose="DINOv2 embed"):
            try:
                return self._forward_batch(images)
            except RuntimeError as exc:
                message = str(exc)
                message_l = message.lower()

                # Missing Metal ops (e.g. upsample_bicubic2d) → CPU, not OOM retry.
                if (
                    self._device.type == "mps"
                    and is_mps_unsupported_op_error(message_l)
                    and not self._mps_cpu_fallback_done
                ):
                    self._fallback_mps_to_cpu(message)
                    return self._extract_batch(images, for_query=for_query)

                is_oom = _is_device_oom_error(self._device.type, message_l)
                if (
                    not is_oom
                    or self._device.type not in ("cuda", "mps")
                    or len(images) <= 1
                ):
                    raise

                logger.warning(
                    "%s OOM on batch of %d views — splitting and retrying.",
                    self._device.type.upper(),
                    len(images),
                )
                if self._device.type == "cuda":
                    torch.cuda.empty_cache()
                mid = len(images) // 2
                left = self._extract_batch(images[:mid], for_query=for_query)
                right = self._extract_batch(images[mid:], for_query=for_query)
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
        *,
        for_query: bool = False,
    ) -> np.ndarray:
        """
        Extract a multi-scale DINOv2 embedding from an already-preprocessed image.

        This is the primary entry point — avoids reloading/resizing the image.
        """
        views = self._generate_views(processed.pil)
        view_embeddings = self._extract_batch(views, for_query=for_query)
        final_embedding = self._fuse_embeddings(view_embeddings)

        logger.debug(
            "Multi-scale DINOv2 embedding: views=%d dimension=%d for_query=%s",
            len(views),
            final_embedding.shape[0],
            for_query,
        )
        return final_embedding

    def extract_batch_from_preprocessed(
        self,
        processed_images: List[PreprocessedImage],
    ) -> List[np.ndarray]:
        """
        Extract embeddings for multiple preprocessed images.

        Processes images in small chunks and releases the inference lock
        between chunks so Search can run while a large folder is indexing.
        """
        if not processed_images:
            return []

        results: List[np.ndarray] = []
        # Keep lock holds short so an active Search is not blocked for hours.
        chunk_size = 2
        for start in range(0, len(processed_images), chunk_size):
            chunk = processed_images[start : start + chunk_size]
            all_views: List[Image.Image] = []
            view_counts: List[int] = []
            for processed in chunk:
                views = self._generate_views(processed.pil)
                view_counts.append(len(views))
                all_views.extend(views)

            view_embeddings = self._extract_batch(all_views, for_query=False)

            offset = 0
            for count in view_counts:
                piece = view_embeddings[offset : offset + count]
                results.append(self._fuse_embeddings(piece))
                offset += count

        logger.debug(
            "Batched DINOv2 embeddings: images=%d (chunked)",
            len(processed_images),
        )
        return results

    def extract(self, image_path: str, *, for_query: bool = False) -> np.ndarray:
        """
        Extract embedding from a file path (loads + preprocesses once).

        Prefer extract_from_preprocessed() when the caller already has
        a PreprocessedImage to avoid duplicate I/O.
        """
        processed = ImagePreprocessor.preprocess(image_path)
        return self.extract_from_preprocessed(processed, for_query=for_query)

    def get_embedding(self, image_path: str) -> np.ndarray:
        """Backward-compatible alias for extract()."""
        return self.extract(image_path)
