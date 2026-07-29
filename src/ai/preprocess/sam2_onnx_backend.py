"""
SAM 2.1 ONNX backend for Precise Crop on Mac Intel + Windows CPU.

Works without transformers Sam2Model / torch>=2.5 — the production Mac Intel
stack (torch 2.2 + transformers<5) can still run accurate Precise Crop via
onnxruntime.

Encoder/decoder exports: vietanhdev/segment-anything-2.1-onnx-models
(Apache-2.0, derived from Meta SAM 2.1).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("tilevision.ai.sam2_onnx")

_BUNDLED_DIRNAME = "sam2.1-hiera-tiny-onnx"
_DEFAULT_HF_REPO = "vietanhdev/segment-anything-2.1-onnx-models"
_DEFAULT_HF_ZIP = "sam2.1_hiera_tiny_20260221.zip"

_encoder: Any = None
_decoder: Any = None
_load_error: str | None = None


def sam2_onnx_enabled() -> bool:
    """ONNX path follows the same Settings / env enable flag as Transformers SAM2."""
    from src.ai.preprocess import sam2_backend

    return sam2_backend.sam2_enabled()


def onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


def resolve_sam2_onnx_dir() -> Path | None:
    """
    Locate encoder/decoder ONNX files.

    Order:
      1. TILEVISION_SAM2_ONNX_DIR
      2. model_weights/sam2.1-hiera-tiny-onnx/
    """
    env_dir = os.environ.get("TILEVISION_SAM2_ONNX_DIR", "").strip()
    candidates: list[Path] = []
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    from src.ai.model_paths import runtime_root

    candidates.append(runtime_root() / "model_weights" / _BUNDLED_DIRNAME)

    for directory in candidates:
        enc, dec = _find_onnx_pair(directory)
        if enc is not None and dec is not None:
            return directory
    return None


def _find_onnx_pair(directory: Path) -> tuple[Path | None, Path | None]:
    if not directory.is_dir():
        return None, None
    enc = next(directory.glob("*.encoder.onnx"), None)
    if enc is None:
        enc = directory / "encoder.onnx"
        if not enc.is_file():
            enc = None
    dec = next(directory.glob("*.decoder.onnx"), None)
    if dec is None:
        dec = directory / "decoder.onnx"
        if not dec.is_file():
            dec = None
    return enc, dec


def sam2_onnx_platform_supported() -> bool:
    """True when onnxruntime is importable (Mac Intel / Windows / Linux CPU)."""
    return onnxruntime_available()


def sam2_onnx_should_run() -> bool:
    """Enabled in Settings/env and onnxruntime present (weights checked at load)."""
    return sam2_onnx_enabled() and sam2_onnx_platform_supported()


def sam2_onnx_status() -> str:
    from src.utils.platform_info import is_mac_intel, is_macos, is_windows

    if is_windows():
        label = "Windows"
    elif is_mac_intel():
        label = "Mac Intel"
    elif is_macos():
        label = "Mac Apple Silicon"
    else:
        label = "Linux"

    if not sam2_onnx_enabled():
        return f"ONNX SAM2 disabled on {label}"
    if not onnxruntime_available():
        return f"ONNX SAM2 unavailable on {label} (install onnxruntime)"
    if _load_error:
        return f"ONNX SAM2 load failed on {label}: {_load_error}"
    if _encoder is not None:
        return f"ONNX SAM2 ready on {label}"
    directory = resolve_sam2_onnx_dir()
    if directory is None:
        return (
            f"ONNX SAM2 missing weights on {label} — "
            "run scripts/download_sam2_onnx_model.py"
        )
    return f"ONNX SAM2 enabled on {label} ({directory})"


def _cpu_providers() -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    preferred = []
    for name in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"):
        if name in available:
            preferred.append(name)
    return preferred or ["CPUExecutionProvider"]


class _SAM2ImageEncoder:
    def __init__(self, path: str) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(path, providers=_cpu_providers())
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.input_names = [item.name for item in inputs]
        self.output_names = [item.name for item in outputs]
        shape = inputs[0].shape
        self.input_height = int(shape[2])
        self.input_width = int(shape[3])

    def __call__(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tensor = self._prepare(image_bgr)
        outputs = self.session.run(self.output_names, {self.input_names[0]: tensor})
        return outputs[0], outputs[1], outputs[2]

    def _prepare(self, image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_width, self.input_height))
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normed = (resized.astype(np.float32) / 255.0 - mean) / std
        return normed.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)


class _SAM2ImageDecoder:
    def __init__(self, path: str, encoder_hw: tuple[int, int]) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(path, providers=_cpu_providers())
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.input_names = [item.name for item in inputs]
        self.output_names = [item.name for item in outputs]
        self.encoder_input_size = encoder_hw
        self.orig_im_size = encoder_hw
        self.scale_factor = 4

    def set_image_size(self, orig_im_size: tuple[int, int]) -> None:
        self.orig_im_size = orig_im_size

    def __call__(
        self,
        image_embed: np.ndarray,
        high_res_feats_0: np.ndarray,
        high_res_feats_1: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        coords, labels = self._prepare_points(point_coords, point_labels)
        num_labels = labels.shape[0]
        mask_input = np.zeros(
            (
                num_labels,
                1,
                self.encoder_input_size[0] // self.scale_factor,
                self.encoder_input_size[1] // self.scale_factor,
            ),
            dtype=np.float32,
        )
        has_mask_input = np.zeros((num_labels,), dtype=np.float32)
        feeds = {
            self.input_names[0]: image_embed,
            self.input_names[1]: high_res_feats_0,
            self.input_names[2]: high_res_feats_1,
            self.input_names[3]: coords,
            self.input_names[4]: labels,
            self.input_names[5]: mask_input,
            self.input_names[6]: has_mask_input,
        }
        outputs = self.session.run(self.output_names, feeds)
        scores = np.asarray(outputs[1]).reshape(-1)
        masks = np.asarray(outputs[0])
        # masks: (1, num_masks, H, W) or similar
        if masks.ndim == 4:
            best = int(np.argmax(scores)) if scores.size else 0
            best_mask = masks[0, best]
        elif masks.ndim == 3:
            best = int(np.argmax(scores)) if scores.size else 0
            best_mask = masks[best]
        else:
            best_mask = masks
        best_mask = cv2.resize(
            best_mask.astype(np.float32),
            (self.orig_im_size[1], self.orig_im_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return best_mask, scores

    def _prepare_points(
        self, point_coords: np.ndarray, point_labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        coords = np.asarray(point_coords, dtype=np.float32)
        labels = np.asarray(point_labels, dtype=np.float32)
        if coords.ndim == 2:
            coords = coords[np.newaxis, ...]
            labels = labels[np.newaxis, ...]
        coords = coords.copy()
        coords[..., 0] = coords[..., 0] / float(self.orig_im_size[1]) * self.encoder_input_size[1]
        coords[..., 1] = coords[..., 1] / float(self.orig_im_size[0]) * self.encoder_input_size[0]
        return coords.astype(np.float32), labels.astype(np.float32)


def load_sam2_onnx() -> tuple[Any, Any]:
    global _encoder, _decoder, _load_error

    if not sam2_onnx_enabled():
        raise RuntimeError("ONNX SAM2 is disabled (Settings / TILEVISION_ENABLE_SAM2).")
    if not onnxruntime_available():
        raise RuntimeError("onnxruntime is not installed.")
    if _encoder is not None and _decoder is not None:
        return _encoder, _decoder

    directory = resolve_sam2_onnx_dir()
    if directory is None:
        raise FileNotFoundError(
            "ONNX SAM2 weights not found. Run: python scripts/download_sam2_onnx_model.py"
        )
    enc_path, dec_path = _find_onnx_pair(directory)
    assert enc_path is not None and dec_path is not None

    try:
        logger.info("Loading ONNX SAM2 from %s", directory)
        encoder = _SAM2ImageEncoder(str(enc_path))
        decoder = _SAM2ImageDecoder(
            str(dec_path), (encoder.input_height, encoder.input_width)
        )
        _encoder = encoder
        _decoder = decoder
        _load_error = None
    except Exception as exc:
        _encoder = None
        _decoder = None
        _load_error = str(exc)
        raise

    return _encoder, _decoder


def segment_tile_mask_onnx(
    image: Image.Image,
    *,
    box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """
    Return a boolean HxW mask using ONNX SAM2 (point + optional box prompt).
    """
    encoder, decoder = load_sam2_onnx()
    rgb = image.convert("RGB")
    width, height = rgb.size
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)

    t0 = time.perf_counter()
    high0, high1, embed = encoder(bgr)
    original_size = (height, width)
    decoder.set_image_size(original_size)

    if box is None:
        points = np.array([[width // 2, height // 2]], dtype=np.float32)
        labels = np.array([1], dtype=np.float32)
    else:
        left, top, right, bottom = box
        # Box corners as SAM2 box prompts (labels 2/3) + center positive point.
        cx = (left + right) // 2
        cy = (top + bottom) // 2
        points = np.array(
            [[left, top], [right, bottom], [cx, cy]],
            dtype=np.float32,
        )
        labels = np.array([2, 3, 1], dtype=np.float32)

    mask_logits, _scores = decoder(embed, high0, high1, points, labels)
    binary = mask_logits > 0.0
    logger.info(
        "ONNX SAM2 mask done in %.2fs (area=%.3f)",
        time.perf_counter() - t0,
        float(binary.mean()),
    )
    return binary.astype(bool)


# Re-export HF ids for the download script.
DEFAULT_ONNX_REPO = _DEFAULT_HF_REPO
DEFAULT_ONNX_ZIP = _DEFAULT_HF_ZIP
BUNDLED_ONNX_DIRNAME = _BUNDLED_DIRNAME
