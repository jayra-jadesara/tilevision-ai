"""
Customer feature parity: Windows ↔ Mac Intel ↔ Mac Apple Silicon.

Hardware acceleration may differ (CUDA / MPS / CPU). Everything a showroom
uses day-to-day — search, Precise Crop backend order, updates, HEIC, timeouts,
query-view caps on CPU — must match across the three client platforms.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.conftest import simulate_platform

# (sys.platform, machine, label, update_key)
CLIENT_PLATFORMS = (
    ("win32", None, "Windows", "windows"),
    ("darwin", "x86_64", "Mac Intel", "macos_intel"),
    ("darwin", "arm64", "Mac Apple Silicon", "macos_arm64"),
)


@pytest.fixture(params=CLIENT_PLATFORMS, ids=lambda p: p[2])
def client_platform(request, monkeypatch):
    platform, machine, label, update_key = request.param
    simulate_platform(monkeypatch, platform, machine=machine)
    return {
        "platform": platform,
        "machine": machine,
        "label": label,
        "update_key": update_key,
    }


def test_update_download_key_matches_client(client_platform):
    from src.utils.update_check import platform_download_key, platform_download_label

    assert platform_download_key() == client_platform["update_key"]
    label = platform_download_label()
    if client_platform["platform"] == "win32":
        assert "Windows" in label
    elif client_platform["machine"] == "x86_64":
        assert "Intel" in label
    else:
        assert "Apple Silicon" in label or "Silicon" in label


def test_search_never_auto_aborts_on_any_client(client_platform):
    from unittest.mock import MagicMock

    from src.presentation.viewmodels.search_viewmodel import (
        SearchViewModel,
        _default_search_timeout_ms,
    )

    assert _default_search_timeout_ms() == 0
    vm = SearchViewModel(use_case=MagicMock())
    assert vm._search_timeout_ms == 0


def test_precise_crop_onnx_is_primary_on_every_client(client_platform, monkeypatch):
    from src.ai.preprocess import sam2_backend
    from src.ai.preprocess.precise_tile_crop import expected_precise_backend

    monkeypatch.setenv("TILEVISION_ENABLE_SAM2", "1")
    sam2_backend.configure_sam2_from_settings(True)
    monkeypatch.setattr(
        "src.ai.preprocess.sam2_onnx_backend.sam2_onnx_should_run",
        lambda: True,
    )
    assert expected_precise_backend() == "sam2"


def test_precise_crop_grabcut_fallback_identical(client_platform, monkeypatch):
    from src.ai.preprocess import sam2_backend
    from src.ai.preprocess.precise_tile_crop import expected_precise_backend

    monkeypatch.setenv("TILEVISION_ENABLE_SAM2", "1")
    monkeypatch.setattr(sam2_backend, "sam2_should_run", lambda: False)
    monkeypatch.setattr(
        "src.ai.preprocess.sam2_onnx_backend.sam2_onnx_should_run",
        lambda: False,
    )
    assert expected_precise_backend() == "grabcut"


def test_cpu_query_views_capped_identically(client_platform, monkeypatch):
    """CPU clients (any OS) share the same multi-crop budget."""
    from src.ai.preprocess.image_preprocessor import ImagePreprocessor
    import src.ai.gpu_info as gpu_info

    monkeypatch.setattr(
        gpu_info,
        "detect_gpu_runtime",
        lambda preference="auto": types.SimpleNamespace(active_device="cpu"),
    )

    assert ImagePreprocessor._capped_query_max_views(3) == 2
    assert ImagePreprocessor._capped_query_max_views(1) == 1


def test_gpu_query_views_uncapped_on_cuda_and_mps(monkeypatch):
    """Windows CUDA and Apple Silicon MPS keep the full multi-crop budget."""
    from src.ai.preprocess.image_preprocessor import ImagePreprocessor
    import src.ai.gpu_info as gpu_info

    for device in ("cuda", "mps"):
        monkeypatch.setattr(
            gpu_info,
            "detect_gpu_runtime",
            lambda preference="auto", d=device: types.SimpleNamespace(active_device=d),
        )
        assert ImagePreprocessor._capped_query_max_views(3) == 3


def test_drop_search_never_invokes_sam2(client_platform, monkeypatch):
    from PIL import Image

    import src.ai.preprocess.fast_tile_crop as fast_tile_crop
    from src.ai.preprocess.image_preprocessor import ImagePreprocessor

    calls = {"precise": 0}

    def _boom(*_a, **_k):
        calls["precise"] += 1
        raise AssertionError("SAM2 must not run on default drop-search")

    monkeypatch.setitem(
        sys.modules,
        "src.ai.preprocess.precise_tile_crop",
        types.SimpleNamespace(precise_isolate_tile=_boom),
    )
    monkeypatch.setattr(
        fast_tile_crop,
        "isolate_tile_region",
        lambda image: types.SimpleNamespace(
            image=image.crop((10, 10, 100, 100)),
            method="opencv",
            confidence=0.7,
        ),
    )
    cropped = ImagePreprocessor._isolate_query_tile(
        Image.new("RGB", (640, 400), color=(120, 110, 100))
    )
    assert calls["precise"] == 0
    assert cropped.size == (90, 90)


def test_heic_extensions_shared_when_registered(client_platform, monkeypatch):
    import src.utils.image_formats as image_formats

    monkeypatch.setattr(image_formats, "_heif_registered", True)
    monkeypatch.setattr(image_formats, "register_optional_image_formats", lambda: None)
    query = image_formats.query_image_extensions()
    indexed = image_formats.supported_image_extensions()
    assert ".heic" in query and ".heif" in query
    assert ".heic" in indexed and ".heif" in indexed
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        assert ext in query
        assert ext in indexed


def test_sam2_onnx_providers_stable_per_os(client_platform, monkeypatch):
    from src.ai.preprocess import sam2_onnx_backend

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.get_available_providers = lambda: [
        "CoreMLExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    providers = sam2_onnx_backend._cpu_providers()
    if client_platform["platform"] == "darwin":
        assert providers == ["CPUExecutionProvider"]
        assert "CoreMLExecutionProvider" not in providers
    else:
        assert providers[0] == "CUDAExecutionProvider"
        assert "CPUExecutionProvider" in providers
        assert "CoreMLExecutionProvider" not in providers


def test_installer_sam2_bundling_identical_for_all_archs(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packaging"))
    from pyinstaller_common import (
        should_bundle_sam2,
        should_bundle_sam2_onnx,
        should_bundle_sam2_transformers,
    )

    monkeypatch.setenv("TILEVISION_BUNDLE_SAM2", "auto")
    monkeypatch.delenv("TILEVISION_BUNDLE_SAM2_TRANSFORMERS", raising=False)

    for arch in (None, "x64", "arm64"):
        assert should_bundle_sam2(macos_arch=arch) is True
        assert should_bundle_sam2_onnx(macos_arch=arch) is True
        # Transformers stays off so Windows == Mac Intel == Mac Silicon packages.
        assert should_bundle_sam2_transformers(macos_arch=arch) is False


def test_platform_labels_cover_all_three_clients(monkeypatch):
    from src.ai.preprocess import sam2_onnx_backend
    from src.utils import platform_info

    expected = {
        ("win32", None): "Windows",
        ("darwin", "x86_64"): "Mac Intel",
        ("darwin", "arm64"): "Mac Apple Silicon",
    }
    for (platform, machine), label in expected.items():
        simulate_platform(monkeypatch, platform, machine=machine)
        status = sam2_onnx_backend.sam2_onnx_status()
        assert label in status
        if platform == "win32":
            assert platform_info.is_windows()
        elif machine == "x86_64":
            assert platform_info.is_mac_intel()
            assert not platform_info.is_apple_silicon()
        else:
            assert platform_info.is_apple_silicon()
            assert not platform_info.is_mac_intel()
