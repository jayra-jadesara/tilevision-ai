"""Tests for experimental precise tile crop (GrabCut + optional SAM2 hook)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.preprocess import sam2_backend
from src.ai.preprocess.precise_tile_crop import precise_isolate_tile, save_precise_tile_crop


def _make_room_like_photo(path: Path) -> None:
    canvas = np.full((420, 900, 3), 40, dtype=np.uint8)
    canvas[180:420, :] = (120, 110, 100)
    rng = np.random.default_rng(11)
    tile = rng.integers(150, 220, size=(220, 220, 3), dtype=np.uint8)
    canvas[160:380, 340:560] = tile
    Image.fromarray(canvas).save(path)


def test_sam2_disabled_by_default(monkeypatch):
    monkeypatch.delenv("TILEVISION_ENABLE_SAM2", raising=False)
    assert sam2_backend.sam2_enabled_by_env() is False
    assert "Disabled" in sam2_backend.sam2_status()


def test_precise_isolate_uses_grabcut_or_fast_without_sam2(tmp_path, monkeypatch):
    monkeypatch.delenv("TILEVISION_ENABLE_SAM2", raising=False)
    path = tmp_path / "room.jpg"
    _make_room_like_photo(path)
    with Image.open(path) as img:
        result = precise_isolate_tile(img.convert("RGB"))

    assert result.method in {"grabcut", "fast_fallback"}
    assert result.image.size[0] > 0
    src_w, src_h = 900, 420
    assert result.image.size[0] * result.image.size[1] <= src_w * src_h


def test_precise_isolate_uses_sam2_when_mocked(tmp_path, monkeypatch):
    monkeypatch.setenv("TILEVISION_ENABLE_SAM2", "1")
    path = tmp_path / "room_sam.jpg"
    _make_room_like_photo(path)

    def _fake_mask(image, box=None):
        width, height = image.size
        mask = np.zeros((height, width), dtype=bool)
        # Central tile-ish rectangle
        mask[160:380, 340:560] = True
        return mask

    monkeypatch.setattr(sam2_backend, "sam2_api_available", lambda: True)
    monkeypatch.setattr(sam2_backend, "sam2_enabled_by_env", lambda: True)
    monkeypatch.setattr(sam2_backend, "segment_tile_mask", _fake_mask)
    monkeypatch.setattr(sam2_backend, "sam2_status", lambda: "Ready (mocked)")

    with Image.open(path) as img:
        result = precise_isolate_tile(img.convert("RGB"))

    assert result.method == "sam2"
    assert result.confidence >= 0.8
    # Should be close to the synthetic tile patch size.
    assert 180 <= result.image.size[0] <= 260
    assert 180 <= result.image.size[1] <= 260


def test_save_precise_tile_crop_writes_jpeg(tmp_path, monkeypatch):
    monkeypatch.delenv("TILEVISION_ENABLE_SAM2", raising=False)
    path = tmp_path / "room_save.jpg"
    _make_room_like_photo(path)
    out_path, result = save_precise_tile_crop(path)
    assert out_path.exists()
    assert "tilevision_crops" in out_path.as_posix()
    assert result.method in {"grabcut", "fast_fallback", "sam2"}


def test_load_sam2_requires_flag(monkeypatch):
    monkeypatch.delenv("TILEVISION_ENABLE_SAM2", raising=False)
    with pytest.raises(RuntimeError, match="disabled"):
        sam2_backend.load_sam2_model()
