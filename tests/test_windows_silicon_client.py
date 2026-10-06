"""Windows + Mac Apple Silicon client polish (parity with Mac Intel)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.mac_client

from PySide6.QtWidgets import QApplication

import src.ai.gpu_info as gpu_info
import src.utils.platform_info as platform_info
from src.presentation.viewmodels.search_viewmodel import (
    SearchViewModel,
    _default_search_timeout_ms,
)
from src.presentation.views.update_dialog import UpdateAvailableDialog
from src.utils.update_check import UpdateInfo, platform_download_key, platform_download_label


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_mac_silicon_search_timeout_disabled_by_default(mac_silicon_platform):
    assert platform_info.is_apple_silicon()
    assert _default_search_timeout_ms() == 0
    use_case = MagicMock()
    vm = SearchViewModel(use_case=use_case)
    assert vm._search_timeout_ms == 0


def test_windows_search_timeout_disabled_by_default(windows_platform):
    assert platform_info.is_windows()
    assert _default_search_timeout_ms() == 0
    use_case = MagicMock()
    vm = SearchViewModel(use_case=use_case)
    assert vm._search_timeout_ms == 0


def test_mac_silicon_sam2_onnx_uses_cpu_provider_only(mac_silicon_platform, monkeypatch):
    from src.ai.preprocess import sam2_onnx_backend

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.get_available_providers = lambda: [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
        "CUDAExecutionProvider",
    ]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    providers = sam2_onnx_backend._cpu_providers()
    assert providers == ["CPUExecutionProvider"]
    assert "CoreMLExecutionProvider" not in providers


def test_windows_sam2_onnx_prefers_cuda_then_cpu(windows_platform, monkeypatch):
    from src.ai.preprocess import sam2_onnx_backend

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.get_available_providers = lambda: [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
        "CoreMLExecutionProvider",
    ]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    providers = sam2_onnx_backend._cpu_providers()
    assert providers[0] == "CUDAExecutionProvider"
    assert "CPUExecutionProvider" in providers
    assert "CoreMLExecutionProvider" not in providers


def test_windows_cpu_caps_query_views(windows_platform, tmp_path, monkeypatch):
    from PIL import Image

    import src.ai.preprocess.fast_tile_crop as fast_tile_crop
    from src.ai.preprocess.image_preprocessor import ImagePreprocessor

    path = tmp_path / "room.jpg"
    Image.new("RGB", (900, 500), color=(170, 160, 150)).save(path)

    monkeypatch.setattr(
        gpu_info,
        "detect_gpu_runtime",
        lambda preference="auto": types.SimpleNamespace(active_device="cpu"),
    )
    monkeypatch.setattr(
        ImagePreprocessor,
        "_looks_like_scene_photo",
        classmethod(lambda cls, img: True),
    )
    monkeypatch.setattr(
        fast_tile_crop,
        "list_tile_region_candidates",
        lambda image, limit=3: [
            types.SimpleNamespace(
                image=Image.new("RGB", (128, 128), color=(i * 40, 80, 100)),
                method=f"cand{i}",
                confidence=0.9,
            )
            for i in range(max(1, int(limit)))
        ],
    )
    views = ImagePreprocessor.prepare_query_views(path, max_views=3)
    assert 1 <= len(views) <= 2


def test_windows_cuda_keeps_full_query_views(windows_platform, tmp_path, monkeypatch):
    from PIL import Image

    import src.ai.preprocess.fast_tile_crop as fast_tile_crop
    from src.ai.preprocess.image_preprocessor import ImagePreprocessor

    path = tmp_path / "room.jpg"
    Image.new("RGB", (900, 500), color=(170, 160, 150)).save(path)

    monkeypatch.setattr(
        gpu_info,
        "detect_gpu_runtime",
        lambda preference="auto": types.SimpleNamespace(active_device="cuda"),
    )
    monkeypatch.setattr(
        ImagePreprocessor,
        "_looks_like_scene_photo",
        classmethod(lambda cls, img: True),
    )
    monkeypatch.setattr(
        fast_tile_crop,
        "list_tile_region_candidates",
        lambda image, limit=3: [
            types.SimpleNamespace(
                image=Image.new("RGB", (128, 128), color=(i * 40, 80, 100)),
                method=f"cand{i}",
                confidence=0.9,
            )
            for i in range(max(1, int(limit)))
        ],
    )
    views = ImagePreprocessor.prepare_query_views(path, max_views=3)
    assert len(views) == 3


def test_mac_silicon_query_views_capped_like_intel(
    mac_silicon_platform, tmp_path, monkeypatch
):
    """Apple Silicon uses the same ≤2 search multi-crop budget as Mac Intel."""
    from PIL import Image

    import src.ai.gpu_info as gpu_info
    import src.ai.preprocess.fast_tile_crop as fast_tile_crop
    from src.ai.preprocess.image_preprocessor import ImagePreprocessor

    path = tmp_path / "room.jpg"
    Image.new("RGB", (900, 500), color=(170, 160, 150)).save(path)

    # MPS available — Mac search budget still matches Intel (≤2).
    monkeypatch.setattr(
        gpu_info,
        "detect_gpu_runtime",
        lambda preference="auto": types.SimpleNamespace(active_device="mps"),
    )
    monkeypatch.setattr(
        ImagePreprocessor,
        "_looks_like_scene_photo",
        classmethod(lambda cls, img: True),
    )
    monkeypatch.setattr(
        fast_tile_crop,
        "list_tile_region_candidates",
        lambda image, limit=3: [
            types.SimpleNamespace(
                image=Image.new("RGB", (128, 128), color=(i * 40, 80, 100)),
                method=f"cand{i}",
                confidence=0.9,
            )
            for i in range(max(1, int(limit)))
        ],
    )
    views = ImagePreprocessor.prepare_query_views(path, max_views=3)
    assert 1 <= len(views) <= 2


def test_mac_silicon_update_key_and_dialog(mac_silicon_platform, qapp):
    assert platform_download_key() == "macos_arm64"
    assert "Apple Silicon" in platform_download_label()
    info = UpdateInfo(
        current_version="1.0.20",
        latest_version="1.0.21",
        release_notes="Silicon polish",
        download_url="https://example.com/TileVisionAI-macOS-AppleSilicon-1.0.21.dmg",
    )
    dialog = UpdateAvailableDialog(info, theme="light", auto_start_download=False)
    dialog.close()


def test_mac_silicon_check_for_updates_uses_silicon_dmg(mac_silicon_platform):
    """Mirror Mac Intel update path — Silicon never receives the Intel DMG."""
    import json
    from unittest.mock import patch

    import src.utils.update_check as update_check

    manifest = {
        "version": "1.0.11",
        "release_notes": "Mac Silicon search + SAM2 parity",
        "downloads": {
            "windows": "https://example.com/setup.exe",
            "macos_intel": "https://example.com/TileVisionAI-macOS-Intel-1.0.11.dmg",
            "macos_arm64": "https://example.com/TileVisionAI-macOS-AppleSilicon-1.0.11.dmg",
        },
    }
    payload = json.dumps(manifest).encode("utf-8")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return payload

    with patch.object(update_check.urllib.request, "urlopen", return_value=_Response()):
        info = update_check.check_for_updates(current_version="1.0.10")

    assert info is not None
    assert info.latest_version == "1.0.11"
    assert "AppleSilicon" in info.download_url
    assert "macOS-Intel" not in info.download_url


def test_mac_silicon_search_does_not_auto_abort(mac_silicon_platform):
    """Same as Mac Intel — never wall-clock abort a working search."""
    from unittest.mock import MagicMock

    from src.presentation.viewmodels.search_viewmodel import (
        SearchViewModel,
        _default_search_timeout_ms,
    )

    assert _default_search_timeout_ms() == 0
    use_case = MagicMock()
    use_case.get_index_health.return_value = MagicMock(
        is_compatible=True, stale_count=0, indexed_count=1
    )
    use_case.get_searchable_count.return_value = 1
    vm = SearchViewModel(use_case=use_case)
    assert vm._search_timeout_ms == 0


def test_windows_update_key_and_dialog(windows_platform, qapp):
    assert platform_download_key() == "windows"
    assert "Windows" in platform_download_label()
    info = UpdateInfo(
        current_version="1.0.20",
        latest_version="1.0.21",
        release_notes="Windows polish",
        download_url="https://example.com/TileVisionAI-Setup-1.0.21.exe",
    )
    dialog = UpdateAvailableDialog(info, theme="light", auto_start_download=False)
    dialog.close()


def test_mac_silicon_never_forced_mps_for_query_preference(mac_silicon_platform, monkeypatch):
    """Apple Silicon may expose MPS, but query path must still be allowed to use CPU."""
    fake_torch = types.SimpleNamespace(
        __version__="2.2.2",
        cuda=types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        version=types.SimpleNamespace(cuda=None),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setattr(gpu_info, "torch", fake_torch)
    monkeypatch.setattr(gpu_info, "detect_display_adapters", lambda: ["Apple M2"])
    monkeypatch.setattr(gpu_info, "has_nvidia_gpu", lambda: False)

    info = gpu_info.detect_gpu_runtime(preference="auto")
    assert platform_info.is_apple_silicon()
    assert info.active_device == "mps"
