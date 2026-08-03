"""Tests for Mac MPS search resilience (unsupported ops → CPU)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.ai.embedder as embedder_module


_MPS_BICUBIC_ERROR = (
    "The operator 'aten::upsample_bicubic2d.out' is not currently "
    "implemented for the MPS device. If you want this op to be added, "
    "set PYTORCH_ENABLE_MPS_FALLBACK=1."
)


def _make_embedder(*, device: str = "mps") -> embedder_module.DINOv2Embedder:
    embedder = embedder_module.DINOv2Embedder.__new__(embedder_module.DINOv2Embedder)
    embedder._device = embedder_module.torch.device(device)
    embedder._device_preference = "auto"
    embedder._runtime = SimpleNamespace(
        active_device=device,
        device_name="Apple M2",
        summary_for_log=lambda: f"{device}",
    )
    embedder._processor = MagicMock()
    embedder._model = MagicMock()
    embedder._mps_cpu_fallback_done = False
    return embedder


def test_is_device_oom_error_ignores_mps_unimplemented_op():
    assert (
        embedder_module._is_device_oom_error(
            "mps",
            _MPS_BICUBIC_ERROR.lower(),
        )
        is False
    )
    assert embedder_module._is_device_oom_error("mps", "mps out of memory") is True
    assert embedder_module._is_device_oom_error("cuda", "cuda error: out of memory") is True


def test_is_device_oom_error_ignores_unsupported_mps_autocast():
    msg = "user specified an unsupported autocast device_type 'mps'"
    assert embedder_module._is_device_oom_error("mps", msg) is False


def test_extract_batch_falls_back_to_cpu_on_unsupported_mps_autocast(monkeypatch):
    """Client Mac Intel log: autocast error was mislabeled MPS OOM then hard-failed."""
    embedder = _make_embedder(device="mps")
    images = [
        Image.new("RGB", (64, 64), color=(10, 20, 30)),
        Image.new("RGB", (64, 64), color=(40, 50, 60)),
        Image.new("RGB", (64, 64), color=(70, 80, 90)),
    ]
    calls = {"n": 0}
    cpu_result = np.ones((3, 1024), dtype=np.float32)

    def _forward(batch):
        calls["n"] += 1
        if embedder._device.type == "mps":
            raise RuntimeError(
                "User specified an unsupported autocast device_type 'mps'"
            )
        return cpu_result[: len(batch)]

    monkeypatch.setattr(embedder, "_forward_batch", _forward)
    monkeypatch.setattr(
        embedder_module,
        "detect_gpu_runtime",
        lambda preference="auto": SimpleNamespace(
            active_device="cpu",
            device_name="",
            summary_for_log=lambda: "cpu",
        ),
    )
    monkeypatch.setattr(embedder_module, "synchronized_inference", lambda **_kwargs: _NullCtx())

    result = embedder._extract_batch(images)

    assert calls["n"] == 2
    assert embedder._device.type == "cpu"
    assert embedder._mps_cpu_fallback_done is True
    assert result.shape[0] == 3


def test_extract_batch_falls_back_to_cpu_on_mps_unimplemented_op(monkeypatch):
    embedder = _make_embedder(device="mps")
    images = [Image.new("RGB", (64, 64), color=(10, 20, 30))]

    calls = {"n": 0}
    cpu_result = np.ones((1, 1024), dtype=np.float32)

    def _forward(batch):
        calls["n"] += 1
        if embedder._device.type == "mps":
            raise RuntimeError(_MPS_BICUBIC_ERROR)
        return cpu_result

    monkeypatch.setattr(embedder, "_forward_batch", _forward)
    monkeypatch.setattr(
        embedder_module,
        "detect_gpu_runtime",
        lambda preference="auto": SimpleNamespace(
            active_device="cpu",
            device_name="",
            summary_for_log=lambda: "cpu",
        ),
    )
    monkeypatch.setattr(embedder_module, "synchronized_inference", lambda **_kwargs: _NullCtx())

    result = embedder._extract_batch(images)

    assert calls["n"] == 2
    assert embedder._device.type == "cpu"
    assert embedder._mps_cpu_fallback_done is True
    assert result is cpu_result


def test_extract_batch_does_not_treat_mps_op_error_as_oom(monkeypatch):
    embedder = _make_embedder(device="mps")
    images = [
        Image.new("RGB", (64, 64), color=(1, 2, 3)),
        Image.new("RGB", (64, 64), color=(4, 5, 6)),
        Image.new("RGB", (64, 64), color=(7, 8, 9)),
    ]

    def _forward(_batch):
        raise RuntimeError(_MPS_BICUBIC_ERROR)

    monkeypatch.setattr(embedder, "_forward_batch", _forward)
    # Force "already fell back" so we exercise the OOM branch path vs raise.
    embedder._mps_cpu_fallback_done = True
    monkeypatch.setattr(embedder_module, "synchronized_inference", lambda **_kwargs: _NullCtx())

    with pytest.raises(RuntimeError, match="upsample_bicubic2d"):
        embedder._extract_batch(images)


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False
