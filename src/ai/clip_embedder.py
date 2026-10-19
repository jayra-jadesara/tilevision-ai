"""
CLIP joint image–text embedder for description search.

Separate from ``DINOv2Embedder`` — different embedding space (512D).
Used only for text search and a parallel FAISS index.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from src.ai.clip_model_paths import (
    CLIP_EMBEDDING_DIM,
    DEFAULT_CLIP_MODEL_ID,
    resolve_clip_model_source,
)
from src.ai.inference_guard import synchronized_inference
from src.ai.preprocess.image_preprocessor import ImagePreprocessor

logger = logging.getLogger("tilevision.ai.clip_embedder")

# Approximate full bundle size for first-use progress (matches measured ~580.7 MB).
_CLIP_BUNDLE_MB_HINT = 581
ProgressCallback = Optional[Callable[[str], None]]


def _dir_bytes(path: Path) -> int:
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _safe_progress(on_progress: ProgressCallback, message: str) -> None:
    if on_progress is None:
        return
    try:
        on_progress(message)
    except Exception:
        pass


class ClipEmbedder:
    """
    OpenAI CLIP ViT-B/32 image + text encoder (transformers).

    Lazy-loads weights on first ``load_model()`` / embed call so customers
    who never use description search pay no startup cost.
    """

    MODEL_ID = DEFAULT_CLIP_MODEL_ID
    EMBEDDING_DIM = CLIP_EMBEDDING_DIM

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_model(self, on_progress: ProgressCallback = None) -> None:
        if self._loaded:
            return
        source, local_only = resolve_clip_model_source()
        if not local_only:
            # First-use path: pull into the bundled weights dir so later
            # sessions / indexing see a local asset (not only the HF cache).
            source = self._download_to_bundled(source, on_progress=on_progress)
            local_only = True
        _safe_progress(on_progress, "Loading description-search model…")
        logger.info(
            "Loading CLIP model from %s (local_files_only=%s)",
            source,
            local_only,
        )
        from transformers import CLIPModel, CLIPProcessor
        import torch

        self._processor = CLIPProcessor.from_pretrained(
            source, local_files_only=local_only
        )
        self._model = CLIPModel.from_pretrained(
            source, local_files_only=local_only
        )
        self._model.eval()
        if torch.cuda.is_available():
            self._device = "cuda"
            self._model.to(self._device)
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self._device = "mps"
            self._model.to(self._device)
        else:
            self._device = "cpu"
        self._loaded = True
        logger.info("CLIP ready on %s (dim=%s)", self._device, self.EMBEDDING_DIM)
        _safe_progress(on_progress, "Description-search model ready")

    @staticmethod
    def _download_to_bundled(
        repo_id: str, on_progress: ProgressCallback = None
    ) -> str:
        from src.ai.clip_model_paths import bundled_clip_model_dir

        out = bundled_clip_model_dir()
        out.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading CLIP weights to %s (first use)…", out)
        _safe_progress(
            on_progress,
            "Downloading description-search model (one-time, ~580 MB)…",
        )

        stop = threading.Event()

        def _progress_ticker() -> None:
            # Keep the search stall watchdog honest and surface MB progress
            # while snapshot_download blocks (can exceed 90s on real links).
            while not stop.wait(2.0):
                downloaded_mb = _dir_bytes(out) / (1024.0 * 1024.0)
                _safe_progress(
                    on_progress,
                    f"Downloading description-search model… "
                    f"{downloaded_mb:.0f} / ~{_CLIP_BUNDLE_MB_HINT} MB",
                )

        ticker: Optional[threading.Thread] = None
        if on_progress is not None:
            ticker = threading.Thread(
                target=_progress_ticker,
                name="clip-download-progress",
                daemon=True,
            )
            ticker.start()

        try:
            from huggingface_hub import snapshot_download

            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(out),
                    local_dir_use_symlinks=False,
                    allow_patterns=[
                        "config.json",
                        "preprocessor_config.json",
                        "tokenizer*",
                        "vocab.json",
                        "merges.txt",
                        "special_tokens_map.json",
                        "model.safetensors",
                        "pytorch_model.bin",
                        "*.json",
                    ],
                    ignore_patterns=[
                        "*.h5",
                        "*.msgpack",
                        "flax_model*",
                        "tf_model*",
                        ".git*",
                        "*.md",
                    ],
                )
            except Exception as exc:
                logger.error("CLIP weight download failed: %s", exc)
                raise RuntimeError(
                    "Could not download the description-search model. "
                    "Check your internet connection and try again when online."
                ) from exc
            weights = list(out.glob("*.safetensors")) + list(out.glob("*.bin"))
            if not weights:
                from huggingface_hub import hf_hub_download

                for name in ("model.safetensors", "pytorch_model.bin"):
                    try:
                        hf_hub_download(
                            repo_id=repo_id, filename=name, local_dir=str(out)
                        )
                        break
                    except Exception as exc:
                        logger.warning(
                            "CLIP weight candidate %s unavailable: %s", name, exc
                        )
        finally:
            stop.set()
            if ticker is not None and ticker.is_alive():
                ticker.join(timeout=1.0)

        return str(out)

    def ensure_loaded(self, on_progress: ProgressCallback = None) -> None:
        self.load_model(on_progress=on_progress)

    def embed_text(self, text: str) -> np.ndarray:
        """Return L2-normalized 512D text embedding."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str] | tuple[str, ...]) -> list[np.ndarray]:
        """
        Return L2-normalized 512D embeddings for each text in one batched
        ``get_text_features`` call (avoids N sequential forward passes for
        prompt-template ensembling).
        """
        cleaned = [(t or "").strip() for t in texts]
        if not cleaned or any(not c for c in cleaned):
            raise ValueError("Description text is empty.")
        self.ensure_loaded()
        import torch

        with synchronized_inference(timeout=120.0, purpose="CLIP text"):
            inputs = self._processor(
                text=cleaned, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self._model.get_text_features(**inputs)
            arr = feats.detach().float().cpu().numpy().astype(np.float32)
        return [self._l2_normalize(arr[i]) for i in range(arr.shape[0])]

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        """Return L2-normalized 512D image embedding (single full view)."""
        self.ensure_loaded()
        import torch

        pil = ImagePreprocessor.to_rgb(ImagePreprocessor.load(Path(image_path)))
        with synchronized_inference(timeout=120.0, purpose="CLIP image"):
            inputs = self._processor(images=pil, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self._model.get_image_features(**inputs)
            vec = feats[0].detach().float().cpu().numpy().astype(np.float32)
        return self._l2_normalize(vec)

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if not np.isfinite(norm) or norm < 1e-8:
            raise RuntimeError("CLIP embedding is empty or not finite.")
        return arr / norm
