"""
Real SAM2 + GrabCut inference smoke tests (Windows / Mac Intel / Mac Silicon paths).

Skipped automatically when experimental SAM2 weights are not present locally.
Download once with:

    python scripts/download_sam2_model.py
"""

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
)
from src.ai.model_paths import runtime_root

_WEIGHTS = runtime_root() / "model_weights" / "sam2.1-hiera-tiny" / "config.json"
_require_weights = pytest.mark.skipif(
    not _WEIGHTS.is_file(),
    reason="SAM2 weights missing — run scripts/download_sam2_model.py",
)


def _make_room_like_photo(path: Path) -> tuple[int, int, int, int]:
    canvas = np.full((420, 900, 3), 40, dtype=np.uint8)
    canvas[180:420, :] = (120, 110, 100)
    rng = np.random.default_rng(11)
    tile = rng.integers(150, 220, size=(220, 220, 3), dtype=np.uint8)
    tile[::20, :] = np.clip(tile[::20, :].astype(int) - 30, 0, 255).astype(np.uint8)
    tile[:, ::20] = np.clip(tile[:, ::20].astype(int) - 30, 0, 255).astype(np.uint8)
    box = (340, 160, 560, 380)
    canvas[box[1] : box[3], box[0] : box[2]] = tile
    Image.fromarray(canvas).save(path)
    return box


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter) / float(union) if union else 0.0


@_require_weights
def test_real_sam2_stack_ready():
    assert sam2_backend.sam2_api_available()
    assert sam2_backend.sam2_platform_supported()


def _label_as_platform(monkeypatch, platform: str, machine: str | None = None) -> None:
    """
    Patch TileVision platform helpers without changing sys.platform.

    Changing sys.platform to win32 on Linux breaks torch/PySide6 (DLL path probes).
    Product routing uses is_windows / is_mac_intel / is_apple_silicon instead.
    """
    import src.utils.platform_info as platform_info

    monkeypatch.setattr(platform_info, "is_windows", lambda: platform == "win32")
    monkeypatch.setattr(platform_info, "is_macos", lambda: platform == "darwin")
    monkeypatch.setattr(platform_info, "is_linux", lambda: platform.startswith("linux"))
    if machine is None:
        monkeypatch.setattr(platform_info, "is_apple_silicon", lambda: False)
        monkeypatch.setattr(platform_info, "is_mac_intel", lambda: False)
    else:
        monkeypatch.setattr(platform_info, "mac_machine", lambda: machine.lower())
        monkeypatch.setattr(
            platform_info,
            "is_apple_silicon",
            lambda: platform == "darwin" and machine.lower() in {"arm64", "aarch64"},
        )
        monkeypatch.setattr(
            platform_info,
            "is_mac_intel",
            lambda: platform == "darwin"
            and machine.lower() not in {"arm64", "aarch64"},
        )


@_require_weights
@pytest.mark.parametrize(
    "platform,machine",
    [
        ("win32", None),
        ("darwin", "x86_64"),
        ("darwin", "arm64"),
    ],
)
def test_real_precise_crop_sam2_high_iou(tmp_path, monkeypatch, platform, machine):
    """Transformers SAM2 fallback still works when ONNX is forced off."""
    if not sam2_backend.sam2_api_available() or not sam2_backend.sam2_platform_supported():
        pytest.skip("SAM2 stack not available in this environment")

    path = tmp_path / "room.jpg"
    gt = _make_room_like_photo(path)

    _label_as_platform(monkeypatch, platform, machine=machine)
    monkeypatch.setenv("TILEVISION_ENABLE_SAM2", "1")
    monkeypatch.setenv("TILEVISION_SAM2_MODEL_DIR", str(_WEIGHTS.parent))
    sam2_backend.configure_sam2_from_settings(True)
    monkeypatch.setattr(
        "src.ai.preprocess.sam2_onnx_backend.sam2_onnx_should_run",
        lambda: False,
    )
    assert expected_precise_backend() == "sam2_transformers"

    with Image.open(path) as img:
        result = precise_isolate_tile(img.convert("RGB"))

    assert result.method == "sam2_transformers", result.detail
    assert _iou(result.box, gt) >= 0.70
    assert result.image.size[0] * result.image.size[1] < 900 * 420 * 0.35
    detail = (result.detail or "") + sam2_backend.sam2_status()
    if platform == "win32":
        assert "Windows" in detail
    elif machine == "x86_64":
        assert "Mac Intel" in detail
    else:
        assert "Apple Silicon" in detail or "Mac" in detail


@_require_weights
def test_real_grabcut_fallback_when_sam2_disabled(tmp_path, monkeypatch):
    path = tmp_path / "room.jpg"
    gt = _make_room_like_photo(path)
    monkeypatch.delenv("TILEVISION_ENABLE_SAM2", raising=False)
    sam2_backend.configure_sam2_from_settings(False)
    monkeypatch.setattr(sam2_backend, "sam2_should_run", lambda: False)
    monkeypatch.setattr(
        "src.ai.preprocess.sam2_onnx_backend.sam2_onnx_should_run",
        lambda: False,
    )

    for platform, machine in (("win32", None), ("darwin", "x86_64"), ("darwin", "arm64")):
        _label_as_platform(monkeypatch, platform, machine=machine)
        with Image.open(path) as img:
            result = precise_isolate_tile(img.convert("RGB"))
        assert result.method in {"grabcut", "fast_fallback"}
        assert _iou(result.box, gt) >= 0.10
        assert result.image.size[0] * result.image.size[1] < 900 * 420
