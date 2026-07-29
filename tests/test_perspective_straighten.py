"""Tests for perspective straighten (query-only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.preprocess.perspective_straighten import straighten_tile_view


def test_straighten_returns_image_for_plain_tile():
    rng = np.random.default_rng(1)
    tile = rng.integers(80, 200, size=(300, 300, 3), dtype=np.uint8)
    image = Image.fromarray(tile)
    out = straighten_tile_view(image)
    assert out.size[0] > 0 and out.size[1] > 0
    assert out.mode == "RGB"


def test_straighten_handles_small_image():
    image = Image.new("RGB", (40, 40), color=(120, 120, 120))
    out = straighten_tile_view(image)
    assert out.size == (40, 40)
