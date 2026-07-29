"""Tests for experimental precise tile crop (GrabCut + optional SAM2 hook)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.preprocess import sam2_backend
from src.ai.preprocess.precise_tile_crop import (
    expected_precise_backend,
    precise_isolate_tile,
    save_precise_tile_crop,
)
from tests.conftest import simulate_platform


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


def test_mac_intel_never_runs_sam2(monkeypatch):
    simulate_platform(monkeypatch, "darwin", machine="x86_64")
    monkeypatch.setenv("TILEVISION_ENABLE_SAM2", "1")
    monkeypatch.setattr(sam2_backend, "sam2_api_available", lambda: True)
    monkeypatch.setattr(sam2_backend, "_torch_version_tuple", lambda: (2, 5, 1))

    assert sam2_backend.sam2_platform_supported() is False
    assert sam2_backend.sam2_should_run() is False
    assert expected_precise_backend() == "grabcut"
    assert "Mac Intel" in sam2_backend.sam2_status()


def test_apple_silicon_can_enable_sam2(monkeypatch):
    simulate_platform(monkeypatch, "darwin", machine="arm64")
    monkeypatch.setenv("TILEVISION_ENABLE_SAM2", "1")
    monkeypatch.setattr(sam2_backend, "sam2_api_available", lambda: True)
    monkeypatch.setattr(sam2_backend, "_torch_version_tuple", lambda: (2, 5, 1))

    assert sam2_backend.sam2_platform_supported() is True
    assert sam2_backend.sam2_should_run() is True
    assert expected_precise_backend() == "sam2"


def test_windows_can_enable_sam2(monkeypatch):
    simulate_platform(monkeypatch, "win32")
    monkeypatch.setenv("TILEVISION_ENABLE_SAM2", "1")
    monkeypatch.setattr(sam2_backend, "sam2_api_available", lambda: True)
    monkeypatch.setattr(sam2_backend, "_torch_version_tuple", lambda: (2, 5, 1))

    assert sam2_backend.sam2_platform_supported() is True
    assert sam2_backend.sam2_should_run() is True
    assert expected_precise_backend() == "sam2"


@pytest.mark.parametrize(
    "platform,machine",
    [
        ("win32", None),
        ("darwin", "x86_64"),
        ("darwin", "arm64"),
    ],
)
def test_precise_crop_works_on_all_major_platforms(tmp_path, monkeypatch, platform, machine):
    simulate_platform(monkeypatch, platform, machine=machine)
    monkeypatch.delenv("TILEVISION_ENABLE_SAM2", raising=False)

    path = tmp_path / f"room_{platform}_{machine or 'na'}.jpg"
    _make_room_like_photo(path)
    with Image.open(path) as img:
        result = precise_isolate_tile(img.convert("RGB"))

    assert result.method in {"grabcut", "fast_fallback"}
    assert result.image.size[0] > 0
    assert result.image.size[1] > 0
    # Must shrink vs full room frame for useful search.
    assert result.image.size[0] * result.image.size[1] < 900 * 420


def test_precise_isolate_uses_sam2_when_mocked(tmp_path, monkeypatch):
    simulate_platform(monkeypatch, "darwin", machine="arm64")
    monkeypatch.setenv("TILEVISION_ENABLE_SAM2", "1")
    path = tmp_path / "room_sam.jpg"
    _make_room_like_photo(path)

    def _fake_mask(image, box=None):
        width, height = image.size
        mask = np.zeros((height, width), dtype=bool)
        mask[160:380, 340:560] = True
        return mask

    monkeypatch.setattr(sam2_backend, "sam2_api_available", lambda: True)
    monkeypatch.setattr(sam2_backend, "_torch_version_tuple", lambda: (2, 5, 1))
    monkeypatch.setattr(sam2_backend, "segment_tile_mask", _fake_mask)
    monkeypatch.setattr(sam2_backend, "sam2_status", lambda: "Ready (mocked)")

    with Image.open(path) as img:
        result = precise_isolate_tile(img.convert("RGB"))

    assert result.method == "sam2"
    assert result.confidence >= 0.8
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
