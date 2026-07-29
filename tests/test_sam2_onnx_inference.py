"""Real ONNX SAM2 inference — Mac Intel / Windows CPU path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.model_paths import runtime_root
from src.ai.preprocess import sam2_backend, sam2_onnx_backend
from src.ai.preprocess.precise_tile_crop import expected_precise_backend, precise_isolate_tile

_ONNX_DIR = runtime_root() / "model_weights" / "sam2.1-hiera-tiny-onnx"
_require_onnx = pytest.mark.skipif(
    not any(_ONNX_DIR.glob("*.encoder.onnx")),
    reason="ONNX SAM2 missing — run scripts/download_sam2_onnx_model.py",
)


def _make_room(path: Path) -> tuple[int, int, int, int]:
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


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = (
        max(0, ax1 - ax0) * max(0, ay1 - ay0)
        + max(0, bx1 - bx0) * max(0, by1 - by0)
        - inter
    )
    return float(inter) / float(union) if union else 0.0


def _label_as(monkeypatch, platform: str, machine: str | None = None) -> None:
    import src.utils.platform_info as platform_info

    monkeypatch.setattr(platform_info, "is_windows", lambda: platform == "win32")
    monkeypatch.setattr(platform_info, "is_macos", lambda: platform == "darwin")
    monkeypatch.setattr(platform_info, "is_linux", lambda: platform.startswith("linux"))
    if machine is None:
        monkeypatch.setattr(platform_info, "is_apple_silicon", lambda: False)
        monkeypatch.setattr(platform_info, "is_mac_intel", lambda: False)
    else:
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


@_require_onnx
def test_onnx_runtime_ready():
    assert sam2_onnx_backend.onnxruntime_available()
    assert sam2_onnx_backend.resolve_sam2_onnx_dir() is not None


@_require_onnx
@pytest.mark.parametrize(
    "platform,machine",
    [
        ("win32", None),
        ("darwin", "x86_64"),
    ],
)
def test_mac_intel_and_windows_use_onnx_precise_crop(
    tmp_path, monkeypatch, platform, machine
):
    """Force Transformers SAM2 off — production Intel/Windows path is ONNX."""
    path = tmp_path / "room.jpg"
    gt = _make_room(path)

    _label_as(monkeypatch, platform, machine)
    monkeypatch.setenv("TILEVISION_ENABLE_SAM2", "1")
    monkeypatch.setenv("TILEVISION_SAM2_ONNX_DIR", str(_ONNX_DIR))
    sam2_backend.configure_sam2_from_settings(True)
    # Simulate production stack without Sam2Model / torch>=2.5.
    monkeypatch.setattr(sam2_backend, "sam2_should_run", lambda: False)

    assert expected_precise_backend() == "sam2_onnx"

    with Image.open(path) as img:
        result = precise_isolate_tile(img.convert("RGB"))

    assert result.method == "sam2_onnx", result.detail
    assert _iou(result.box, gt) >= 0.55
    assert result.image.size[0] * result.image.size[1] < 900 * 420 * 0.45
    detail = result.detail + sam2_onnx_backend.sam2_onnx_status()
    if platform == "win32":
        assert "Windows" in detail
    else:
        assert "Mac Intel" in detail
