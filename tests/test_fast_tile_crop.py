"""Tests for fast OpenCV tile-region auto-crop."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.preprocess.fast_tile_crop import isolate_tile_region
from src.ai.preprocess.image_preprocessor import ImagePreprocessor, TARGET_SIZE


def _make_room_like_photo(path: Path) -> None:
    """Wide scene with a textured near-square tile patch in the center."""
    canvas = np.full((420, 900, 3), 40, dtype=np.uint8)  # dark furniture/walls
    # Floor-ish background
    canvas[180:420, :] = (120, 110, 100)
    # Central tile with speckled texture
    rng = np.random.default_rng(7)
    tile = rng.integers(150, 220, size=(220, 220, 3), dtype=np.uint8)
    canvas[160:380, 340:560] = tile
    Image.fromarray(canvas).save(path)


def test_isolate_tile_region_shrinks_room_photo(tmp_path):
    path = tmp_path / "room.jpg"
    _make_room_like_photo(path)
    with Image.open(path) as img:
        source = img.convert("RGB")
        result = isolate_tile_region(source)

    sw, sh = source.size
    cw, ch = result.image.size
    assert cw * ch < sw * sh
    assert result.confidence > 0.0
    assert result.method in {"contour", "texture", "center_fallback"}


def test_isolate_tile_region_is_fast_on_cpu(tmp_path):
    path = tmp_path / "room_speed.jpg"
    _make_room_like_photo(path)
    with Image.open(path) as img:
        source = img.convert("RGB")

    started = time.perf_counter()
    for _ in range(5):
        isolate_tile_region(source)
    elapsed = time.perf_counter() - started
    # 5 runs should stay well under a second on CPU.
    assert elapsed < 1.0


def test_clean_square_tile_skips_scene_auto_crop_path(tmp_path):
    path = tmp_path / "clean_tile.jpg"
    rng = np.random.default_rng(3)
    tile = rng.integers(80, 180, size=(400, 400, 3), dtype=np.uint8)
    Image.fromarray(tile).save(path)

    with Image.open(path) as img:
        assert not ImagePreprocessor._looks_like_scene_photo(img.convert("RGB"))

    processed = ImagePreprocessor.preprocess_for_query(path)
    assert processed.pil.size == (TARGET_SIZE, TARGET_SIZE)


def test_preprocess_for_query_handles_room_photo(tmp_path):
    path = tmp_path / "room_query.jpg"
    _make_room_like_photo(path)
    processed = ImagePreprocessor.preprocess_for_query(path)
    assert processed.pil.size == (TARGET_SIZE, TARGET_SIZE)
    assert processed.width == 900
    assert processed.height == 420


def test_save_auto_tile_crop_writes_temp_jpeg(tmp_path):
    from src.ai.preprocess.fast_tile_crop import save_auto_tile_crop

    path = tmp_path / "room_save.jpg"
    _make_room_like_photo(path)
    out_path, result = save_auto_tile_crop(path)
    assert out_path.exists()
    assert out_path.suffix.lower() == ".jpg"
    assert "tilevision_crops" in out_path.as_posix()
    assert result.image.size[0] > 0
    with Image.open(out_path) as saved:
        assert saved.size == result.image.size
