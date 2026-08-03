"""Tests for GPU runtime detection."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.ai.gpu_info as gpu_info


def test_cpu_wheel_reports_install_hint(monkeypatch):
    fake_torch = SimpleNamespace(
        __version__="2.13.0+cpu",
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        version=SimpleNamespace(cuda=None),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(gpu_info, "torch", fake_torch)
    monkeypatch.setattr(gpu_info, "detect_display_adapters", lambda: [])
    monkeypatch.setattr(gpu_info, "has_nvidia_gpu", lambda: False)

    info = gpu_info.detect_gpu_runtime(preference="auto")

    assert info.active_device == "cpu"
    assert "CPU-only PyTorch" in info.cpu_fallback_reason


def test_cuda_auto_selects_gpu(monkeypatch):
    fake_torch = SimpleNamespace(
        __version__="2.5.1+cu124",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda _i: "NVIDIA Test GPU",
            get_device_properties=lambda _i: SimpleNamespace(total_memory=8 * 1024 ** 3),
        ),
        version=SimpleNamespace(cuda="12.4"),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(gpu_info, "torch", fake_torch)
    monkeypatch.setattr(gpu_info, "detect_display_adapters", lambda: [])
    monkeypatch.setattr(gpu_info, "has_nvidia_gpu", lambda: False)

    info = gpu_info.detect_gpu_runtime(preference="auto")

    assert info.using_gpu
    assert info.device_name == "NVIDIA Test GPU"
    assert "NVIDIA Test GPU" in info.summary_for_ui()


def test_non_nvidia_adapter_message(monkeypatch):
    fake_torch = SimpleNamespace(
        __version__="2.13.0+cpu",
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        version=SimpleNamespace(cuda=None),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(gpu_info, "torch", fake_torch)
    monkeypatch.setattr(
        gpu_info,
        "detect_display_adapters",
        lambda: ["AMD Radeon R5 M330", "Intel(R) HD Graphics 520"],
    )
    monkeypatch.setattr(gpu_info, "has_nvidia_gpu", lambda: False)

    info = gpu_info.detect_gpu_runtime(preference="auto")

    assert "no NVIDIA GPU" in info.cpu_fallback_reason


def test_forced_cpu_even_when_cuda_available(monkeypatch):
    fake_torch = SimpleNamespace(
        __version__="2.5.1+cu124",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda _i: "NVIDIA Test GPU",
            get_device_properties=lambda _i: SimpleNamespace(total_memory=8 * 1024 ** 3),
        ),
        version=SimpleNamespace(cuda="12.4"),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr(gpu_info, "torch", fake_torch)
    monkeypatch.setattr(gpu_info, "detect_display_adapters", lambda: [])
    monkeypatch.setattr(gpu_info, "has_nvidia_gpu", lambda: False)

    info = gpu_info.detect_gpu_runtime(preference="cpu")

    assert info.active_device == "cpu"
    assert "forced" in info.cpu_fallback_reason.lower()


def test_mps_auto_selects_on_macos(monkeypatch):
    fake_torch = SimpleNamespace(
        __version__="2.5.1",
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        version=SimpleNamespace(cuda=None),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setattr(gpu_info, "torch", fake_torch)
    monkeypatch.setattr(gpu_info, "_mps_available", lambda: True)
    monkeypatch.setattr(gpu_info, "_mps_device_name", lambda: "Apple M2")
    monkeypatch.setattr(gpu_info, "detect_display_adapters", lambda: [])

    info = gpu_info.detect_gpu_runtime(preference="auto")

    assert info.active_device == "mps"
    assert info.using_gpu
    assert "Apple GPU" in info.summary_for_ui()


def test_mps_blocked_on_mac_intel_even_if_torch_reports_mps(monkeypatch):
    import src.utils.platform_info as platform_info

    fake_torch = SimpleNamespace(
        __version__="2.2.2",
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        version=SimpleNamespace(cuda=None),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setattr(gpu_info, "torch", fake_torch)
    monkeypatch.setattr(platform_info, "is_macos", lambda: True)
    monkeypatch.setattr(platform_info, "mac_machine", lambda: "x86_64")
    monkeypatch.setattr(platform_info, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(platform_info, "is_mac_intel", lambda: True)
    monkeypatch.setattr(gpu_info, "detect_display_adapters", lambda: ["Intel Iris"])
    monkeypatch.setattr(gpu_info, "has_nvidia_gpu", lambda: False)

    info = gpu_info.detect_gpu_runtime(preference="auto")
    assert info.active_device == "cpu"
    assert "Intel Mac" in info.cpu_fallback_reason


def test_mps_autocast_supported_false_when_autocast_rejects_mps(monkeypatch):
    def _raise_autocast(*_args, **_kwargs):
        raise RuntimeError("unsupported autocast device_type 'mps'")

    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        autocast=_raise_autocast,
    )
    monkeypatch.setattr(gpu_info, "torch", fake_torch)
    monkeypatch.setattr(gpu_info, "_mps_available", lambda: True)
    monkeypatch.setattr(gpu_info, "_MPS_AUTOCAST_SUPPORTED", None)

    assert gpu_info.mps_autocast_supported() is False


def test_mps_autocast_supported_true_when_autocast_works(monkeypatch):
    from contextlib import contextmanager

    @contextmanager
    def _ok_autocast(*_args, **_kwargs):
        yield

    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        autocast=_ok_autocast,
    )
    monkeypatch.setattr(gpu_info, "torch", fake_torch)
    monkeypatch.setattr(gpu_info, "_mps_available", lambda: True)
    monkeypatch.setattr(gpu_info, "_MPS_AUTOCAST_SUPPORTED", None)

    assert gpu_info.mps_autocast_supported() is True


def test_configure_mps_fallback_sets_env_on_darwin(monkeypatch):
    monkeypatch.setattr(gpu_info.sys, "platform", "darwin")
    monkeypatch.delenv(gpu_info._MPS_FALLBACK_ENV, raising=False)

    assert gpu_info.configure_mps_fallback() is True
    assert os.environ.get(gpu_info._MPS_FALLBACK_ENV) == "1"


def test_configure_mps_fallback_noop_on_linux(monkeypatch):
    monkeypatch.setattr(gpu_info.sys, "platform", "linux")
    monkeypatch.delenv(gpu_info._MPS_FALLBACK_ENV, raising=False)

    assert gpu_info.configure_mps_fallback() is False
    assert os.environ.get(gpu_info._MPS_FALLBACK_ENV) is None


def test_is_mps_unsupported_op_error_detects_upsample_bicubic2d():
    message = (
        "The operator 'aten::upsample_bicubic2d.out' is not currently "
        "implemented for the MPS device."
    )
    assert gpu_info.is_mps_unsupported_op_error(message) is True
    assert gpu_info.is_mps_unsupported_op_error("CUDA out of memory") is False


def test_is_mps_unsupported_op_error_detects_autocast_rejection():
    assert (
        gpu_info.is_mps_unsupported_op_error(
            "User specified an unsupported autocast device_type 'mps'"
        )
        is True
    )
    assert gpu_info.is_mps_unsupported_op_error("unsupported autocast device_type cuda") is False
