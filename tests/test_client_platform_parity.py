"""
Windows / Mac Intel / Mac Apple Silicon feature-parity contracts.

Locks in identical customer-facing behavior across the three shipped clients
unless a genuine hardware limitation applies (documented in the PR audit).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.conftest import simulate_platform

# (sys.platform, machine) — cover Windows + both Mac arches.
CLIENT_PLATFORMS = (
    ("win32", None),
    ("darwin", "x86_64"),
    ("darwin", "arm64"),
)


@pytest.fixture(params=CLIENT_PLATFORMS, ids=lambda p: f"{p[0]}-{p[1] or 'na'}")
def client_platform(request, monkeypatch):
    platform, machine = request.param
    simulate_platform(monkeypatch, platform, machine=machine)
    return {"platform": platform, "machine": machine}


def test_mac_search_view_budget_matches_intel_and_silicon(client_platform, monkeypatch):
    """Mac Intel + Mac Silicon (+ Windows CPU) share the ≤2 search multi-crop budget."""
    from src.ai.preprocess.image_preprocessor import ImagePreprocessor
    import src.ai.gpu_info as gpu_info

    device = "cpu"
    if client_platform["machine"] == "arm64":
        device = "mps"
    monkeypatch.setattr(
        gpu_info,
        "detect_gpu_runtime",
        lambda preference="auto", d=device: types.SimpleNamespace(active_device=d),
    )

    if client_platform["platform"] == "darwin" or client_platform["platform"] == "win32":
        # Windows CUDA is covered separately; here non-CUDA / Mac stay ≤2.
        if client_platform["platform"] == "win32":
            monkeypatch.setattr(
                gpu_info,
                "detect_gpu_runtime",
                lambda preference="auto": types.SimpleNamespace(active_device="cpu"),
            )
        assert ImagePreprocessor._capped_query_max_views(3) == 2
        assert ImagePreprocessor._capped_query_max_views(1) == 1


def test_windows_cuda_keeps_full_query_views_budget(monkeypatch):
    from src.ai.preprocess.image_preprocessor import ImagePreprocessor
    import src.ai.gpu_info as gpu_info

    simulate_platform(monkeypatch, "win32")
    monkeypatch.setattr(
        gpu_info,
        "detect_gpu_runtime",
        lambda preference="auto": types.SimpleNamespace(active_device="cuda"),
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


def test_sam2_onnx_providers_identical_on_both_mac_arches(client_platform, monkeypatch):
    """Mac Intel and Mac Silicon both force CPUExecutionProvider (no CoreML)."""
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


def test_sam2_onnx_weights_resolver_is_arch_agnostic(client_platform, monkeypatch, tmp_path):
    """Same weight directory / filenames on Intel and Silicon — no arch branch."""
    from src.ai.preprocess import sam2_onnx_backend

    root = tmp_path / "sam2.1-hiera-tiny-onnx"
    root.mkdir()
    (root / "encoder.onnx").write_bytes(b"enc")
    (root / "decoder.onnx").write_bytes(b"dec")
    monkeypatch.setenv("TILEVISION_SAM2_ONNX_DIR", str(root))

    resolved = sam2_onnx_backend.resolve_sam2_onnx_dir()
    assert resolved == root
    assert sam2_onnx_backend.sam2_onnx_platform_supported() in (True, False)


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
