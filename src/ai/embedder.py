"""
DINOv2 embedder module for TileVision AI.

Uses Meta DINOv2 with a batched multi-scale strategy for catalogue indexing:

1. Full tile image   (global context)
2. Center crop       (large region, ~65%)
3. Detail crop       (fine pattern region, ~40%)

Query / drop-search uses a **single full-image view** so Mac Intel / Windows CPU
clients return tile results quickly (multi-crop OpenCV already covers room photos).

All index views are embedded then fused with fixed weights into a 1024D
L2-normalized vector compatible with FAISS.

DINOv2 Large: 1024 dimensions
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import threading
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModel

from src.ai.models import PreprocessedImage
from src.ai.inference_guard import (
    DEFAULT_INDEX_LOCK_TIMEOUT_S,
    DEFAULT_SEARCH_LOCK_TIMEOUT_S,
    interactive_cpu_thread_count,
    is_warmup_compute,
    restore_interactive_torch_threads,
    search_priority_active,
    synchronized_inference,
    wait_while_search_priority,
)
from src.ai.gpu_info import (
    DevicePreference,
    configure_mps_fallback,
    detect_gpu_runtime,
    is_mps_unsupported_op_error,
    mps_autocast_supported,
)
from src.ai.preprocess.image_preprocessor import ImagePreprocessor, TARGET_SIZE

logger = logging.getLogger("tilevision.ai.embedder")

# Weighted fusion of multi-scale views.  Global dominates; detail
# boosts fine-grained pattern discrimination without overpowering semantics.
_VIEW_WEIGHTS: Tuple[float, ...] = (0.50, 0.30, 0.20)

# Query-time MPS hang watchdog.
#
# Working DINOv2-large forwards on Apple Silicon MPS are typically well under
# ~2s for a single (or ≤2) query view; Mac Intel CPU warm-up logs in this
# codebase are usually hundreds of ms to a few seconds. 20s is ~10× margin
# over a slow-but-working MPS forward so legitimate searches are not
# false-positive timed out. Override with TILEVISION_MPS_QUERY_TIMEOUT_S after
# measuring on real Silicon hardware (see scripts/validate_mac_silicon_mps.py).
_DEFAULT_MPS_QUERY_TIMEOUT_S = 20.0
_MPS_QUERY_TIMEOUT_ENV = "TILEVISION_MPS_QUERY_TIMEOUT_S"


def mps_query_forward_timeout_s() -> float:
    """Bounded wait for a query-time MPS forward before CPU hang-fallback."""
    raw = os.environ.get(_MPS_QUERY_TIMEOUT_ENV, "").strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            logger.warning(
                "Ignoring invalid %s=%r — using default %.1fs",
                _MPS_QUERY_TIMEOUT_ENV,
                raw,
                _DEFAULT_MPS_QUERY_TIMEOUT_S,
            )
    return _DEFAULT_MPS_QUERY_TIMEOUT_S


def _is_device_oom_error(device_type: str, message: str) -> bool:
    """True only for genuine out-of-memory failures (not missing MPS ops)."""
    text = (message or "").lower()
    # Client log (Intel Mac): autocast rejection was wrongly treated as
    # "MPS OOM" → split/retry death spiral. Never classify as OOM.
    if "unsupported autocast" in text or is_mps_unsupported_op_error(text):
        return False
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

    def __init__(
        self,
        *,
        device_preference: DevicePreference = "auto",
        pooling: str = "cls",
    ) -> None:
        if pooling not in ("cls", "mean_patch"):
            raise ValueError(
                f"Invalid pooling {pooling!r}: expected 'cls' or 'mean_patch'"
            )
        configure_mps_fallback()
        self._device_preference: DevicePreference = device_preference
        self._pooling = pooling
        self._runtime = detect_gpu_runtime(preference=device_preference)
        self._device = torch.device(self._runtime.active_device)
        self._processor = None
        self._model = None
        self._mps_cpu_fallback_done = False
        self._query_path_warmed = False
        # Serializes load / MPS→CPU reload so background query-warmup cannot
        # race catalogue indexing (meta-tensor crash after hang-watchdog).
        self._model_load_lock = threading.RLock()

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
        with self._model_load_lock:
            if self._model is not None:
                return
            self._load_model_unlocked()

    def _load_model_unlocked(self) -> None:
        """Load DINOv2 if missing; caller must hold ``_model_load_lock``."""
        if self._model is not None:
            return
        self._force_reload_model_unlocked()

    def _force_reload_model_unlocked(
        self,
        *,
        target_device: torch.device | None = None,
    ) -> None:
        """
        Build a fresh DINOv2 and swap it in.

        Caller must hold ``_model_load_lock``. Does **not** null ``_model``
        before the new module is ready — hang-watchdog reload can race
        catalogue indexing, and ``self._model = None`` caused
        ``'NoneType' object is not callable`` (Release Validation macos-15).

        When ``target_device`` is set (MPS→CPU watchdog), the new module is
        built on that device and ``self._device`` is published in the same
        critical section as the model swap — never flip ``_device`` to CPU
        while weights are still on MPS (empty FAISS: ``input(cpu)`` /
        ``weight(mps:0)``).
        """
        logger.info("Loading DINOv2 model...")

        from src.ai.model_paths import resolve_dinov2_model_source

        model_source, local_only = resolve_dinov2_model_source()
        logger.info(
            "DINOv2 source: %s (%s)",
            model_source,
            "offline/local" if local_only else "Hugging Face hub",
        )

        device = target_device if target_device is not None else self._device

        processor = AutoImageProcessor.from_pretrained(
            model_source,
            local_files_only=local_only,
        )
        # low_cpu_mem_usage=False avoids meta-tensor init where .to(device)
        # can raise "Cannot copy out of meta tensor" when two loads race or
        # transformers defaults delay weight materialization.
        model = AutoModel.from_pretrained(
            model_source,
            local_files_only=local_only,
            low_cpu_mem_usage=False,
        )
        model.to(device)
        model.eval()

        # Atomic publish: model + processor (+ device when relocating).
        self._processor = processor
        self._model = model
        if target_device is not None:
            self._device = device

        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            logger.info(
                "CUDA GPU: %s (%.1f GB VRAM)",
                self._runtime.device_name,
                self._runtime.vram_gb or 0.0,
            )
        elif device.type == "mps":
            configure_mps_fallback()
            logger.info("Apple GPU (MPS): %s", self._runtime.device_name)
        else:
            from src.utils.platform_info import is_mac_intel

            # macOS Intel: OpenMP oversubscription inside worker threads hangs
            # search forever. Keep a single intra-op thread on that platform.
            if is_mac_intel():
                thread_count = 1
                try:
                    torch.set_num_interop_threads(1)
                except Exception:
                    pass
            else:
                thread_count = min(8, os.cpu_count() or 4)
            torch.set_num_threads(thread_count)
            logger.info("CPU inference threads: %d", thread_count)

        logger.info("DINOv2 model loaded successfully.")

    @staticmethod
    def dummy_query_view(size: int | None = None) -> PreprocessedImage:
        """Letterboxed dummy view matching crop-tool query input shape (518²)."""
        edge = int(size or TARGET_SIZE)
        image = Image.new("RGB", (edge, edge), color=(128, 132, 140))
        arr = np.asarray(image, dtype=np.uint8)
        gray = np.mean(arr, axis=2).astype(np.uint8)
        return PreprocessedImage(
            pil=image,
            rgb=arr,
            bgr=arr[:, :, ::-1].copy(),
            gray=gray,
            width=edge,
            height=edge,
        )

    def warmup_query_inference(
        self,
        *,
        shapes: tuple[int, ...] = (1,),
    ) -> dict[str, float]:
        """
        Prime crop-tool query embed shapes. Logs n=1 and n=2 separately.

        Windows oneDNN compiles each batch shape independently — n=1 then
        n=2 at startup was ~2× the original 44s first-click cost. Default
        is n=1 only (Auto / Precise Crop). n=2 is opt-in via ``shapes``.

        Aborts remaining shapes if a user search has claimed priority.
        """
        import time as _time

        from src.ai.inference_guard import search_priority_active

        timings: dict[str, float] = {}
        if self._query_path_warmed and 1 in shapes and 2 not in shapes:
            return timings
        if search_priority_active():
            logger.info("Query-path warm-up skipped — search already running")
            return timings

        self.load_model()
        dummy = self.dummy_query_view()

        if 1 in shapes:
            t0 = _time.perf_counter()
            self.extract_query_views_batch([dummy])
            timings["n1_ms"] = (_time.perf_counter() - t0) * 1000.0
            logger.info("Query-path warm-up n=1: %.0f ms", timings["n1_ms"])

        if 2 in shapes:
            if search_priority_active():
                logger.info("Query-path warm-up n=2: skipped — search requested")
            else:
                t0 = _time.perf_counter()
                self.extract_query_views_batch([dummy, dummy])
                timings["n2_ms"] = (_time.perf_counter() - t0) * 1000.0
                logger.info("Query-path warm-up n=2: %.0f ms", timings["n2_ms"])
        else:
            logger.info(
                "Query-path warm-up n=2: skipped "
                "(first 2-view Manual Crop may pay a one-time cost)"
            )

        self._query_path_warmed = True
        return timings

    def _fallback_mps_to_cpu(
        self,
        reason: str,
        *,
        cause: str = "unsupported_op",
    ) -> None:
        """
        Switch DINOv2 to CPU after an MPS failure.

        ``cause`` distinguishes log wording:
          - ``unsupported_op`` — Metal raised (reactive path)
          - ``timeout`` — query watchdog fired (silent hang; no exception)

        Holds ``_model_load_lock`` for the whole device swap + reload so a
        concurrent catalogue-index ``load_model()`` / ``_forward_batch``
        cannot observe ``_device=cpu`` with weights still on MPS (Release
        Validation macos-15: empty FAISS after
        ``input(device='cpu') and weight(device='mps:0')``).
        """
        with self._model_load_lock:
            if self._mps_cpu_fallback_done and self._device.type == "cpu":
                if self._model is None:
                    self._load_model_unlocked()
                return
            short = reason.splitlines()[0][:160]
            if cause == "timeout":
                logger.warning(
                    "MPS query forward timed out — switching DINOv2 to CPU so "
                    "search continues (hang watchdog). (%s)",
                    short,
                )
            else:
                logger.warning(
                    "MPS operator unavailable — switching DINOv2 to CPU so "
                    "search continues. (%s)",
                    short,
                )
            cpu = torch.device("cpu")
            self._device_preference = "cpu"
            self._runtime = detect_gpu_runtime(preference="cpu")
            from src.utils.platform_info import is_mac_intel

            thread_count = 1 if is_mac_intel() else min(8, os.cpu_count() or 4)
            torch.set_num_threads(thread_count)
            if is_mac_intel():
                try:
                    torch.set_num_interop_threads(1)
                except Exception:
                    pass

            if cause == "timeout":
                # Hung MPS worker may still own the old module. Build a fresh
                # CPU copy and publish device+model together — never set
                # ``_device=cpu`` while ``_model`` is still on MPS.
                self._force_reload_model_unlocked(target_device=cpu)
            elif self._model is not None:
                self._model.to(cpu)
                self._model.eval()
                self._device = cpu
            else:
                self._device = cpu
                self._load_model_unlocked()
            self._mps_cpu_fallback_done = True

    def _run_model_forward(
        self,
        inputs: dict,
        *,
        model: object | None = None,
        device: torch.device | None = None,
    ) -> object:
        """Run DINOv2 forward pass with autocast only when the device supports it."""
        model = self._model if model is None else model
        device = self._device if device is None else device
        if device.type == "cuda":
            with torch.autocast(device_type="cuda"):
                return model(**inputs)
        if device.type == "mps":
            if mps_autocast_supported():
                with torch.autocast(device_type="mps"):
                    return model(**inputs)
            logger.debug("MPS autocast unavailable — running float32 inference on MPS")
            return model(**inputs)
        return model(**inputs)

    def _ensure_index_torch_threads(self, *, views: int, purpose: str) -> int:
        """
        Re-assert interactive torch thread budget before an index forward.

        Logs the live ``torch.get_num_threads()`` so customer logs can confirm
        whether a rebuild ran at 1 thread (warmup-cap leak) or the full budget.
        """
        if self._device.type != "cpu":
            return -1
        restored = restore_interactive_torch_threads()
        try:
            current = int(torch.get_num_threads())
        except Exception:
            current = -1
        target = interactive_cpu_thread_count()
        logger.info(
            "DINOv2 index forward prep: purpose=%s views=%d torch_threads=%s "
            "interactive_target=%s restored=%s",
            purpose,
            views,
            current,
            target,
            restored,
        )
        if current > 0 and current < target:
            logger.warning(
                "DINOv2 index forward still below interactive thread budget "
                "(%s < %s) — retrying restore",
                current,
                target,
            )
            restore_interactive_torch_threads()
            try:
                current = int(torch.get_num_threads())
            except Exception:
                pass
        return current

    def _forward_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Single DINOv2 forward pass."""
        # Snapshot under the load lock so a concurrent MPS→CPU fallback cannot
        # pair CPU inputs with MPS weights (or vice versa) mid-forward.
        with self._model_load_lock:
            processor = self._processor
            model = self._model
            device = self._device
        if processor is None or model is None:
            self.load_model()
            with self._model_load_lock:
                processor = self._processor
                model = self._model
                device = self._device

        inputs = processor(images=images, return_tensors="pt")
        inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self._run_model_forward(inputs, model=model, device=device)

        hidden = outputs.last_hidden_state
        if self._pooling == "mean_patch":
            embeddings = (
                hidden[:, 1:]
                .mean(dim=1)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        else:
            embeddings = (
                hidden[:, 0]
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        return embeddings / norms

    def _forward_batch_with_mps_query_watchdog(
        self,
        images: List[Image.Image],
        *,
        timeout_s: float,
    ) -> np.ndarray:
        """
        Run a query forward on MPS inside a bounded-time worker.

        Silent Metal hangs never raise — only a join timeout can detect them.
        On timeout the caller falls back to CPU (see ``_extract_batch``).
        Uses ``shutdown(wait=False)`` so a stuck worker cannot block the UI
        thread forever (the orphaned worker is abandoned with the old module).
        """
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(self._forward_batch, images)
            return future.result(timeout=timeout_s)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _generate_views(
        image: Image.Image,
        *,
        for_query: bool = False,
    ) -> List[Image.Image]:
        """
        Build embedding views from a preprocessed PIL image.

        Query/search: single full image only (fast path — critical on Mac CPU).
        Indexing: global + center + detail multi-scale for catalogue quality.
        """
        image = image.convert("RGB")
        if for_query:
            return [image]

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

        # Keep index + query on the same device (MPS/CUDA/CPU) so cosine ranks
        # match. Query-time MPS uses a hang watchdog (silent Metal hangs never
        # raise). Reactive unsupported-op fallback below is unchanged.
        use_mps_query_watchdog = (
            for_query
            and self._device.type == "mps"
            and not self._mps_cpu_fallback_done
        )
        mps_timeout_s = (
            mps_query_forward_timeout_s() if use_mps_query_watchdog else 0.0
        )

        def _run_forward() -> np.ndarray:
            if use_mps_query_watchdog:
                logger.info(
                    "DINOv2 query forward on MPS with hang watchdog "
                    "(timeout=%.1fs views=%d)",
                    mps_timeout_s,
                    len(images),
                )
                return self._forward_batch_with_mps_query_watchdog(
                    images,
                    timeout_s=mps_timeout_s,
                )
            return self._forward_batch(images)

        def _fallback_after_mps_timeout() -> np.ndarray:
            self._fallback_mps_to_cpu(
                f"MPS query forward exceeded {mps_timeout_s:.1f}s "
                f"(TILEVISION_MPS_QUERY_TIMEOUT_S / default "
                f"{_DEFAULT_MPS_QUERY_TIMEOUT_S:.1f}s)",
                cause="timeout",
            )
            return self._extract_batch(images, for_query=for_query)

        def _fallback_after_mps_op_error(message: str) -> np.ndarray | None:
            message_l = message.lower()
            if (
                self._device.type == "mps"
                and is_mps_unsupported_op_error(message_l)
                and not self._mps_cpu_fallback_done
            ):
                self._fallback_mps_to_cpu(message, cause="unsupported_op")
                return self._extract_batch(images, for_query=for_query)
            return None

        if is_warmup_compute():
            logger.info(
                "DINOv2 warmup forward (no inference lock, torch_threads=%s)",
                torch.get_num_threads() if hasattr(torch, "get_num_threads") else "?",
            )
            try:
                return _run_forward()
            except concurrent.futures.TimeoutError:
                return _fallback_after_mps_timeout()
            except (RuntimeError, ValueError) as exc:
                retried = _fallback_after_mps_op_error(str(exc))
                if retried is not None:
                    return retried
                raise

        # If Search is waiting, indexing must not start another long forward.
        if not for_query and search_priority_active():
            wait_while_search_priority(max_wait_s=180.0)

        with synchronized_inference(timeout=lock_timeout, purpose="DINOv2 embed"):
            try:
                return _run_forward()
            except concurrent.futures.TimeoutError:
                return _fallback_after_mps_timeout()
            except (RuntimeError, ValueError) as exc:
                message = str(exc)
                retried = _fallback_after_mps_op_error(message)
                if retried is not None:
                    return retried

                message_l = message.lower()
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
        Extract a DINOv2 embedding from an already-preprocessed image.

        Query: single-view (fast). Index: multi-scale views fused in **one**
        batched forward pass (not N serial single-image calls). Search can
        still interrupt between images / view-chunks via
        ``wait_while_search_priority`` before the forward.
        """
        views = self._generate_views(processed.pil, for_query=for_query)

        if not for_query:
            wait_while_search_priority()
            self._ensure_index_torch_threads(
                views=len(views),
                purpose="extract_from_preprocessed",
            )

        view_embeddings = self._extract_batch(views, for_query=for_query)
        final_embedding = self._fuse_embeddings(view_embeddings)

        logger.debug(
            "DINOv2 embedding: views=%d dimension=%d for_query=%s forward_batch=%d",
            len(views),
            final_embedding.shape[0],
            for_query,
            len(views),
        )
        return final_embedding

    def extract_query_views_batch(
        self,
        processed_views: List[PreprocessedImage],
    ) -> List[np.ndarray]:
        """
        Embed multiple letterboxed query views in one batched forward when possible.

        Each ``PreprocessedImage`` is already at TARGET_SIZE (518). Query path
        uses a single DINO scale per view — batching N views is ~one forward pass
        instead of N sequential lock acquisitions on CPU.
        """
        if not processed_views:
            return []
        if len(processed_views) == 1:
            return [
                self.extract_from_preprocessed(processed_views[0], for_query=True)
            ]
        pils = [view.pil for view in processed_views]
        batch = self._extract_batch(pils, for_query=True)
        return [
            np.asarray(batch[i], dtype=np.float32) for i in range(batch.shape[0])
        ]

    # Max PIL views per DINOv2 forward during catalogue indexing. Keeps peak
    # RAM bounded on CPU-only showroom PCs while still forming a real batch
    # (multiple images × multi-scale views) instead of 1-image serial calls.
    _INDEX_VIEW_FORWARD_CHUNK = 24

    def extract_batch_from_preprocessed(
        self,
        processed_images: List[PreprocessedImage],
    ) -> List[np.ndarray]:
        """
        Extract embeddings for multiple preprocessed catalogue images.

        Builds every multi-scale view up front, then runs real batched
        ``_forward_batch`` calls (chunked). Previously this looped
        one-image-at-a-time with three serial single-view forwards each —
        the dominant cost on CPU-only full rebuilds (~6–8s/image).
        """
        if not processed_images:
            return []

        if len(processed_images) == 1:
            return [
                self.extract_from_preprocessed(processed_images[0], for_query=False)
            ]

        all_views: List[Image.Image] = []
        views_per_image: List[int] = []
        for processed in processed_images:
            views = self._generate_views(processed.pil, for_query=False)
            views_per_image.append(len(views))
            all_views.extend(views)

        chunk = max(1, int(os.environ.get(
            "TILEVISION_INDEX_VIEW_BATCH",
            str(self._INDEX_VIEW_FORWARD_CHUNK),
        )))
        pieces: List[np.ndarray] = []
        forward_calls = 0
        thread_samples: List[int] = []
        forward_ms: List[float] = []
        import time as _time

        for start in range(0, len(all_views), chunk):
            wait_while_search_priority()
            batch_views = all_views[start : start + chunk]
            threads_now = self._ensure_index_torch_threads(
                views=len(batch_views),
                purpose="extract_batch_from_preprocessed",
            )
            if threads_now > 0:
                thread_samples.append(threads_now)
            t_fwd = _time.perf_counter()
            pieces.append(self._extract_batch(batch_views, for_query=False))
            forward_ms.append((_time.perf_counter() - t_fwd) * 1000.0)
            forward_calls += 1

        stacked = np.vstack(pieces) if len(pieces) > 1 else pieces[0]

        results: List[np.ndarray] = []
        offset = 0
        for n_views in views_per_image:
            fused = self._fuse_embeddings(stacked[offset : offset + n_views])
            results.append(fused)
            offset += n_views

        logger.info(
            "DINOv2 index batch: images=%d views=%d forward_calls=%d "
            "views_per_forward~%d torch_threads=%s forward_ms=%s "
            "(was %d serial single-view calls)",
            len(processed_images),
            len(all_views),
            forward_calls,
            min(chunk, len(all_views)),
            thread_samples if thread_samples else "?",
            [round(ms, 1) for ms in forward_ms],
            len(all_views),
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
