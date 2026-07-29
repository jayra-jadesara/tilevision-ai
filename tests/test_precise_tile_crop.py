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
    sam2_backend.configure_sam2_from_settings(False)
    assert sam2_backend.sam2_enabled() is False


def test_sam2_enabled_via_settings_toggle(monkeypatch):
    monkeypatch.delenv("TILEVISION_ENABLE_SAM2", raising=False)
    sam2_backend.configure_sam2_from_settings(True)
    assert sam2_backend.sam2_enabled_by_settings() is True
    assert sam2_backend.sam2_enabled() is True
    sam2_backend.configure_sam2_from_settings(False)
    assert sam2_backend.sam2_enabled() is False


def test_mac_intel_onnx_is_primary_shared_path(monkeypatch):
    simulate_platform(monkeypatch, "darwin", machine="x86_64")
    monkeypatch.setenv("TILEVISION_ENABLE_SAM2", "1")
    sam2_backend.configure_sam2_from_settings(True)
    monkeypatch.setattr(
        "src.ai.preprocess.sam2_onnx_backend.sam2_onnx_should_run",
        lambda: True,
    )
    assert expected_precise_backend() == "sam2"


def test_windows_and_mac_silicon_share_same_onnx_primary(monkeypatch):
    monkeypatch.setenv("TILEVISION_ENABLE_SAM2", "1")
    sam2_backend.configure_sam2_from_settings(True)
    monkeypatch.setattr(
        "src.ai.preprocess.sam2_onnx_backend.sam2_onnx_should_run",
        lambda: True,
    )
    for platform, machine in (("win32", None), ("darwin", "arm64"), ("darwin", "x86_64")):
        simulate_platform(monkeypatch, platform, machine=machine)
        assert expected_precise_backend() == "sam2"


def test_mac_intel_falls_back_when_all_sam2_unavailable(monkeypatch):
    simulate_platform(monkeypatch, "darwin", machine="x86_64")
    monkeypatch.setenv("TILEVISION_ENABLE_SAM2", "1")
    monkeypatch.setattr(sam2_backend, "sam2_api_available", lambda: False)
    monkeypatch.setattr(
        "src.ai.preprocess.sam2_onnx_backend.sam2_onnx_should_run",
        lambda: False,
    )

    assert sam2_backend.sam2_platform_supported() is False
    assert expected_precise_backend() == "grabcut"


def test_transformers_is_secondary_when_onnx_off(monkeypatch):
    simulate_platform(monkeypatch, "win32")
    monkeypatch.setenv("TILEVISION_ENABLE_SAM2", "1")
    sam2_backend.configure_sam2_from_settings(True)
    monkeypatch.setattr(
        "src.ai.preprocess.sam2_onnx_backend.sam2_onnx_should_run",
        lambda: False,
    )
    monkeypatch.setattr(sam2_backend, "sam2_api_available", lambda: True)
    monkeypatch.setattr(sam2_backend, "_torch_version_tuple", lambda: (2, 5, 1))
    assert expected_precise_backend() == "sam2_transformers"


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
    sam2_backend.configure_sam2_from_settings(False)
    # Force GrabCut path for this cross-platform smoke test.
    monkeypatch.setattr(sam2_backend, "sam2_should_run", lambda: False)
    monkeypatch.setattr(
        "src.ai.preprocess.sam2_onnx_backend.sam2_onnx_should_run",
        lambda: False,
    )

    path = tmp_path / f"room_{platform}_{machine or 'na'}.jpg"
    _make_room_like_photo(path)
    with Image.open(path) as img:
        result = precise_isolate_tile(img.convert("RGB"))

    assert result.method in {"grabcut", "fast_fallback"}
    assert result.image.size[0] > 0
    assert result.image.size[1] > 0
    assert result.image.size[0] * result.image.size[1] < 900 * 420


def test_precise_isolate_uses_onnx_sam2_on_windows_and_mac(tmp_path, monkeypatch):
    path = tmp_path / "room_sam.jpg"
    _make_room_like_photo(path)

    def _fake_mask(image, box=None):
        width, height = image.size
        mask = np.zeros((height, width), dtype=bool)
        mask[160:380, 340:560] = True
        return mask

    for platform, machine in (("win32", None), ("darwin", "x86_64"), ("darwin", "arm64")):
        simulate_platform(monkeypatch, platform, machine=machine)
        monkeypatch.setenv("TILEVISION_ENABLE_SAM2", "1")
        monkeypatch.setattr(
            "src.ai.preprocess.sam2_onnx_backend.sam2_onnx_should_run",
            lambda: True,
        )
        monkeypatch.setattr(
            "src.ai.preprocess.sam2_onnx_backend.segment_tile_mask_onnx",
            _fake_mask,
        )
        monkeypatch.setattr(
            "src.ai.preprocess.sam2_onnx_backend.sam2_onnx_status",
            lambda: "ONNX SAM2 ready (mocked)",
        )

        with Image.open(path) as img:
            result = precise_isolate_tile(img.convert("RGB"))
        assert result.method == "sam2", platform
        assert "ONNX" in result.detail


def test_save_precise_tile_crop_writes_jpeg(tmp_path, monkeypatch):
    monkeypatch.delenv("TILEVISION_ENABLE_SAM2", raising=False)
    monkeypatch.setattr(
        "src.ai.preprocess.sam2_onnx_backend.sam2_onnx_should_run",
        lambda: False,
    )
    path = tmp_path / "room_save.jpg"
    _make_room_like_photo(path)
    out_path, result = save_precise_tile_crop(path)
    assert out_path.exists()
    assert "tilevision_crops" in out_path.as_posix()
    assert result.method in {"grabcut", "fast_fallback", "sam2", "sam2_transformers"}


def test_load_sam2_requires_enable(monkeypatch):
    monkeypatch.delenv("TILEVISION_ENABLE_SAM2", raising=False)
    sam2_backend.configure_sam2_from_settings(False)
    with pytest.raises(RuntimeError, match="disabled"):
        sam2_backend.load_sam2_model()
