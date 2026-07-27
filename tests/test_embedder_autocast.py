"""Tests for DINOv2 embedder device/autocast handling."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.ai.embedder as embedder_module


def test_mps_forward_skips_unsupported_autocast(monkeypatch):
    embedder = embedder_module.DINOv2Embedder.__new__(embedder_module.DINOv2Embedder)
    embedder._device = embedder_module.torch.device("mps")
    embedder._model = MagicMock(return_value=SimpleNamespace(last_hidden_state=MagicMock()))
    embedder._model.return_value.last_hidden_state.__getitem__.return_value = MagicMock()

    monkeypatch.setattr(embedder_module, "mps_autocast_supported", lambda: False)

    calls = {"autocast": 0}

    def _autocast(*_args, **_kwargs):
        calls["autocast"] += 1
        raise AssertionError("MPS autocast should not be used")

    monkeypatch.setattr(embedder_module.torch, "autocast", _autocast)

    embedder._run_model_forward({"pixel_values": MagicMock()})
    embedder._model.assert_called_once()
    assert calls["autocast"] == 0
