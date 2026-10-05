#!/usr/bin/env python3
"""
Profile evidence for catalogue indexing DINOv2 batching (Task 1).

Does NOT download DINOv2 weights. It instruments the embedder call shape so we
can prove whether indexing uses real batched ``_forward_batch`` calls or
serial single-view forwards — the dominant cost on CPU-only full rebuilds.

Usage:
  PYTHONPATH=. python scripts/profile_index_batching.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Lightweight stubs so this script runs without heavy deps installed
# (mirrors tests/conftest.py). Real DINOv2 timing still needs a showroom PC.
import types

if "transformers" not in sys.modules:
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoImageProcessor = object
    fake_tf.AutoModel = object
    sys.modules["transformers"] = fake_tf

if "PySide6" not in sys.modules:
    qtcore = types.ModuleType("PySide6.QtCore")

    class _Qt:
        class AspectRatioMode:
            KeepAspectRatio = 1

        class TransformationMode:
            SmoothTransformation = 1

    class _QSize:
        def __init__(self, *args):
            self.args = args

    qtcore.QSize = _QSize
    qtcore.Qt = _Qt
    qtgui = types.ModuleType("PySide6.QtGui")

    class _QIcon:
        def __init__(self, *args, **kwargs):
            pass

    class _QPixmap:
        def __init__(self, *args, **kwargs):
            pass

        def isNull(self):
            return True

        def scaled(self, *args, **kwargs):
            return self

    qtgui.QIcon = _QIcon
    qtgui.QPixmap = _QPixmap
    root = types.ModuleType("PySide6")
    root.QtCore = qtcore
    root.QtGui = qtgui
    sys.modules["PySide6"] = root
    sys.modules["PySide6.QtCore"] = qtcore
    sys.modules["PySide6.QtGui"] = qtgui

from src.ai.embedder import DINOv2Embedder
from src.ai.models import PreprocessedImage


def _fake_processed(color=(120, 80, 40)) -> PreprocessedImage:
    pil = Image.new("RGB", (518, 518), color=color)
    arr = np.asarray(pil, dtype=np.uint8)
    return PreprocessedImage(
        pil=pil,
        rgb=arr,
        bgr=arr[:, :, ::-1].copy(),
        gray=np.mean(arr, axis=2).astype(np.uint8),
        width=518,
        height=518,
    )


def _instrumented_embedder() -> tuple[DINOv2Embedder, dict]:
    embedder = DINOv2Embedder(device_preference="cpu")
    embedder._model = object()
    stats = {"calls": [], "images_total": 0}

    def fake_forward(images):
        n = len(images)
        stats["calls"].append(n)
        # Cheap stand-in for a forward to keep the script offline-friendly.
        time.sleep(0.002 * n)
        return np.ones((n, 1024), dtype=np.float32)

    embedder._forward_batch = fake_forward  # type: ignore[method-assign]
    return embedder, stats


def main() -> int:
    embedder, stats = _instrumented_embedder()
    images = [_fake_processed((i * 10, 40, 80)) for i in range(12)]

    t0 = time.perf_counter()
    out = embedder.extract_batch_from_preprocessed(images)
    elapsed = time.perf_counter() - t0

    total_views = sum(stats["calls"])
    print("=== Index batching profile (instrumented _forward_batch) ===")
    print(f"images:              {len(images)}")
    print(f"embeddings returned: {len(out)}")
    print(f"forward call sizes:  {stats['calls']}")
    print(f"total views scored:  {total_views}")
    print(f"forward calls:       {len(stats['calls'])}")
    print(f"wall time:           {elapsed * 1000:.1f} ms")
    print()
    print("Evidence:")
    if stats["calls"] == [1] * (len(images) * 3):
        print("  FAIL: still serial single-view forwards (old path).")
        return 1
    if len(stats["calls"]) == 1 and stats["calls"][0] == len(images) * 3:
        print(
            "  OK: one real batched forward for all multi-scale views "
            f"({stats['calls'][0]} views)."
        )
        return 0
    print(
        "  OK: chunked real batched forwards "
        f"(calls={stats['calls']}, not {len(images) * 3} serial singles)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
