"""Tests for GPU runtime detection."""

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
    )
    monkeypatch.setattr(gpu_info, "torch", fake_torch)
    monkeypatch.setattr(gpu_info, "_detect_windows_graphics", lambda: [])

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
    )
    monkeypatch.setattr(gpu_info, "torch", fake_torch)
    monkeypatch.setattr(gpu_info, "_detect_windows_graphics", lambda: [])

    info = gpu_info.detect_gpu_runtime(preference="auto")

    assert info.using_gpu
    assert info.device_name == "NVIDIA Test GPU"
    assert "NVIDIA Test GPU" in info.summary_for_ui()


def test_non_nvidia_adapter_message(monkeypatch):
    fake_torch = SimpleNamespace(
        __version__="2.13.0+cpu",
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
        version=SimpleNamespace(cuda=None),
    )
    monkeypatch.setattr(gpu_info, "torch", fake_torch)
    monkeypatch.setattr(
        gpu_info,
        "_detect_windows_graphics",
        lambda: ["AMD Radeon R5 M330", "Intel(R) HD Graphics 520"],
    )

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
    )
    monkeypatch.setattr(gpu_info, "torch", fake_torch)
    monkeypatch.setattr(gpu_info, "_detect_windows_graphics", lambda: [])

    info = gpu_info.detect_gpu_runtime(preference="cpu")

    assert info.active_device == "cpu"
    assert "forced" in info.cpu_fallback_reason.lower()
