"""Tests for Mac MPS search resilience (unsupported ops + hang watchdog → CPU)."""

from __future__ import annotations

import sys
import threading
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
    embedder._model_load_lock = threading.RLock()
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


def test_extract_batch_query_keeps_mps_when_forward_returns(monkeypatch):
    """Working MPS query stays on Metal (same device as catalogue index)."""
    embedder = _make_embedder(device="mps")
    images = [Image.new("RGB", (64, 64), color=(10, 20, 30))]
    cpu_result = np.ones((1, 1024), dtype=np.float32)
    calls = {"n": 0}

    def _forward(_batch):
        calls["n"] += 1
        assert embedder._device.type == "mps"
        return cpu_result

    monkeypatch.setattr(embedder, "_forward_batch", _forward)
    monkeypatch.setattr(embedder_module, "synchronized_inference", lambda **_kwargs: _NullCtx())
    monkeypatch.setattr(embedder_module, "is_warmup_compute", lambda: False)
    monkeypatch.setattr(embedder_module, "search_priority_active", lambda: False)

    result = embedder._extract_batch(images, for_query=True)

    assert calls["n"] == 1
    assert embedder._device.type == "mps"
    assert embedder._mps_cpu_fallback_done is False
    assert result is cpu_result


def test_extract_batch_query_hang_watchdog_falls_back_to_cpu(monkeypatch):
    """
    Silent MPS hang (no exception) must trip the watchdog, switch to CPU,
    and still return a query result.
    """
    embedder = _make_embedder(device="mps")
    images = [Image.new("RGB", (64, 64), color=(10, 20, 30))]
    cpu_result = np.ones((1, 1024), dtype=np.float32)
    release_hang = threading.Event()
    calls = {"mps": 0, "cpu": 0}
    load_calls = {"n": 0}

    def _forward(_batch):
        if embedder._device.type == "mps":
            calls["mps"] += 1
            # Block past the watchdog timeout — never raises.
            release_hang.wait(timeout=30.0)
            return cpu_result
        calls["cpu"] += 1
        return cpu_result

    def _load_model_unlocked():
        load_calls["n"] += 1
        embedder._model = MagicMock()
        embedder._processor = MagicMock()

    monkeypatch.setenv(embedder_module._MPS_QUERY_TIMEOUT_ENV, "0.2")
    monkeypatch.setattr(embedder, "_forward_batch", _forward)
    monkeypatch.setattr(embedder, "_load_model_unlocked", _load_model_unlocked)
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
    monkeypatch.setattr(embedder_module, "is_warmup_compute", lambda: False)
    monkeypatch.setattr(embedder_module, "search_priority_active", lambda: False)

    try:
        result = embedder._extract_batch(images, for_query=True)
    finally:
        release_hang.set()

    assert calls["mps"] == 1
    assert calls["cpu"] == 1
    assert load_calls["n"] == 1
    assert embedder._device.type == "cpu"
    assert embedder._mps_cpu_fallback_done is True
    assert result is cpu_result


def test_mps_query_timeout_env_override(monkeypatch):
    monkeypatch.setenv(embedder_module._MPS_QUERY_TIMEOUT_ENV, "12.5")
    assert embedder_module.mps_query_forward_timeout_s() == 12.5
    monkeypatch.delenv(embedder_module._MPS_QUERY_TIMEOUT_ENV, raising=False)
    assert (
        embedder_module.mps_query_forward_timeout_s()
        == embedder_module._DEFAULT_MPS_QUERY_TIMEOUT_S
    )


def test_timeout_fallback_serializes_concurrent_load_model(monkeypatch):
    """
    Hang-watchdog reload must not interleave with a concurrent load_model
    (indexing vs warmup) — that race produced meta-tensor empty FAISS on
    Release Validation macos-15.
    """
    embedder = _make_embedder(device="mps")
    embedder._model = MagicMock()
    started = threading.Event()
    release_load = threading.Event()
    loads = {"n": 0, "concurrent": 0}
    in_flight = {"n": 0}

    def _slow_load_unlocked():
        loads["n"] += 1
        in_flight["n"] += 1
        if in_flight["n"] > 1:
            loads["concurrent"] += 1
        started.set()
        assert release_load.wait(timeout=5.0)
        in_flight["n"] -= 1
        embedder._model = MagicMock()
        embedder._processor = MagicMock()

    monkeypatch.setattr(embedder, "_load_model_unlocked", _slow_load_unlocked)
    monkeypatch.setattr(
        embedder_module,
        "detect_gpu_runtime",
        lambda preference="auto": SimpleNamespace(
            active_device="cpu",
            device_name="",
            summary_for_log=lambda: "cpu",
        ),
    )

    def _index_load():
        started.wait(timeout=5.0)
        embedder.load_model()

    indexer = threading.Thread(target=_index_load, name="index-load")
    indexer.start()
    # Warmup hang-watchdog path: abandon MPS module and reload on CPU.
    fallbacker = threading.Thread(
        target=lambda: embedder._fallback_mps_to_cpu("test timeout", cause="timeout"),
        name="fallback",
    )
    fallbacker.start()
    assert started.wait(timeout=5.0)
    # Give the indexer a chance to pile in while reload is in progress.
    threading.Event().wait(0.05)
    release_load.set()
    fallbacker.join(timeout=5.0)
    indexer.join(timeout=5.0)

    assert loads["concurrent"] == 0
    assert loads["n"] >= 1
    assert embedder._device.type == "cpu"
    assert embedder._model is not None


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False
