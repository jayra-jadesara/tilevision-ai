"""Query DINOv2 must stay fast — single-view for search, multi-scale for index."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.embedder import DINOv2Embedder
from src.ai.models import PreprocessedImage


def _fake_processed(size: int = 224) -> PreprocessedImage:
    image = Image.new("RGB", (size, size), color=(120, 130, 140))
    arr = np.asarray(image, dtype=np.uint8)
    gray = np.mean(arr, axis=2).astype(np.uint8)
    return PreprocessedImage(
        pil=image,
        rgb=arr,
        bgr=arr[:, :, ::-1].copy(),
        gray=gray,
        width=size,
        height=size,
    )


def test_query_generate_views_is_single_image():
    image = Image.new("RGB", (256, 256), color=(10, 20, 30))
    query_views = DINOv2Embedder._generate_views(image, for_query=True)
    index_views = DINOv2Embedder._generate_views(image, for_query=False)
    assert len(query_views) == 1
    assert len(index_views) == 3


def test_query_extract_uses_one_forward(monkeypatch):
    embedder = DINOv2Embedder(device_preference="cpu")
    embedder._model = object()  # skip load
    calls = {"batches": []}

    def fake_batch(images, *, for_query=False):
        calls["batches"].append((len(images), for_query))
        return np.ones((len(images), 1024), dtype=np.float32)

    monkeypatch.setattr(embedder, "_extract_batch", fake_batch)
    out = embedder.extract_from_preprocessed(_fake_processed(), for_query=True)
    assert out.shape == (1024,)
    assert calls["batches"] == [(1, True)]


def test_index_extract_yields_between_views(monkeypatch):
    embedder = DINOv2Embedder(device_preference="cpu")
    embedder._model = object()
    calls = {"batches": [], "waits": 0}

    def fake_batch(images, *, for_query=False):
        calls["batches"].append((len(images), for_query))
        return np.ones((len(images), 1024), dtype=np.float32)

    monkeypatch.setattr(embedder, "_extract_batch", fake_batch)
    monkeypatch.setattr(
        "src.ai.embedder.wait_while_search_priority",
        lambda *a, **k: calls.__setitem__("waits", calls["waits"] + 1),
    )
    out = embedder.extract_from_preprocessed(_fake_processed(), for_query=False)
    assert out.shape == (1024,)
    assert calls["batches"] == [(1, False), (1, False), (1, False)]
    assert calls["waits"] == 3
