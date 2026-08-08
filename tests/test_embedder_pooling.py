"""Regression tests for DINOv2Embedder pooling modes."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai import embedder as embedder_module


def _make_embedder(pooling: str) -> embedder_module.DINOv2Embedder:
    embedder = embedder_module.DINOv2Embedder.__new__(embedder_module.DINOv2Embedder)
    embedder._device = MagicMock(type="cpu")
    embedder._pooling = pooling
    embedder._processor = MagicMock()
    embedder._model = MagicMock()
    return embedder


def _fake_hidden(batch: int = 1, tokens: int = 5, dim: int = 1024):
    import torch

    data = torch.arange(batch * tokens * dim, dtype=torch.float32).reshape(
        batch,
        tokens,
        dim,
    )
    return SimpleNamespace(last_hidden_state=data)


def test_default_pooling_is_cls():
    embedder = embedder_module.DINOv2Embedder(device_preference="cpu")
    assert embedder._pooling == "cls"


def test_cls_pooling_matches_legacy_behavior():
    embedder = _make_embedder("cls")
    embedder._processor.return_value = {"pixel_values": MagicMock()}
    embedder._run_model_forward = MagicMock(return_value=_fake_hidden())

    out = embedder._forward_batch([MagicMock()])
    legacy = _fake_hidden().last_hidden_state[:, 0].cpu().numpy().astype(np.float32)
    legacy /= np.linalg.norm(legacy, axis=1, keepdims=True) + 1e-8
    np.testing.assert_allclose(out, legacy)


def test_mean_patch_pooling_differs_from_cls_and_is_normalized():
    hidden = _fake_hidden()
    cls = _make_embedder("cls")
    cls._processor.return_value = {"pixel_values": MagicMock()}
    cls._run_model_forward = MagicMock(return_value=hidden)

    mean = _make_embedder("mean_patch")
    mean._processor.return_value = {"pixel_values": MagicMock()}
    mean._run_model_forward = MagicMock(return_value=hidden)

    cls_out = cls._forward_batch([MagicMock()])
    mean_out = mean._forward_batch([MagicMock()])

    assert cls_out.shape == (1, 1024)
    assert mean_out.shape == (1, 1024)
    assert not np.allclose(cls_out, mean_out)
    assert pytest.approx(1.0) == float(np.linalg.norm(mean_out[0]))


def test_invalid_pooling_rejected():
    with pytest.raises(ValueError, match="Invalid pooling"):
        embedder_module.DINOv2Embedder(pooling="attention")
