"""Tests for optional SAM2 installer bundling — identical on Mac and Windows."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packaging"))

from pyinstaller_common import (  # noqa: E402
    collect_datas,
    should_bundle_sam2,
    should_bundle_sam2_onnx,
    should_bundle_sam2_transformers,
)


@pytest.fixture(autouse=True)
def _clear_bundle_flag(monkeypatch):
    monkeypatch.delenv("TILEVISION_BUNDLE_SAM2", raising=False)
    monkeypatch.delenv("TILEVISION_BUNDLE_SAM2_TRANSFORMERS", raising=False)
    monkeypatch.delenv("MACOS_BUILD_ARCH", raising=False)


def test_should_bundle_sam2_default_off():
    assert should_bundle_sam2() is False
    assert should_bundle_sam2_onnx() is False
    assert should_bundle_sam2_transformers() is False


@pytest.mark.parametrize("arch", ["x64", "arm64", None])
def test_auto_bundles_onnx_identically_on_all_mac_archs_and_windows(monkeypatch, arch):
    monkeypatch.setenv("TILEVISION_BUNDLE_SAM2", "auto")
    if arch:
        monkeypatch.setenv("MACOS_BUILD_ARCH", arch)
    assert should_bundle_sam2(macos_arch=arch) is True
    assert should_bundle_sam2_onnx(macos_arch=arch) is True
    # Transformers off by default so Mac Intel == Windows == Silicon.
    assert should_bundle_sam2_transformers(macos_arch=arch) is False


def test_transformers_bundle_requires_explicit_flag(monkeypatch):
    monkeypatch.setenv("TILEVISION_BUNDLE_SAM2", "auto")
    monkeypatch.setenv("TILEVISION_BUNDLE_SAM2_TRANSFORMERS", "1")
    assert should_bundle_sam2_transformers(macos_arch="x64") is True
    assert should_bundle_sam2_transformers(macos_arch="arm64") is True


def test_collect_datas_includes_onnx_for_intel_and_windows(tmp_path, monkeypatch):
    monkeypatch.setenv("TILEVISION_BUNDLE_SAM2", "auto")
    monkeypatch.setenv("MACOS_BUILD_ARCH", "x64")
    root = tmp_path
    (root / "src" / "config").mkdir(parents=True)
    (root / "src" / "config" / "default_config.json").write_text("{}")
    onnx = root / "model_weights" / "sam2.1-hiera-tiny-onnx"
    onnx.mkdir(parents=True)
    (onnx / "sam2.1_hiera_tiny.encoder.onnx").write_bytes(b"x")
    (onnx / "sam2.1_hiera_tiny.decoder.onnx").write_bytes(b"y")

    datas = collect_datas(root)
    dests = [dest for _src, dest in datas]
    assert any("sam2.1-hiera-tiny-onnx" in dest for dest in dests)


def test_collect_datas_omits_sam2_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("TILEVISION_BUNDLE_SAM2", raising=False)
    root = tmp_path
    (root / "src" / "config").mkdir(parents=True)
    (root / "src" / "config" / "default_config.json").write_text("{}")
    onnx = root / "model_weights" / "sam2.1-hiera-tiny-onnx"
    onnx.mkdir(parents=True)
    (onnx / "sam2.1_hiera_tiny.encoder.onnx").write_bytes(b"x")

    datas = collect_datas(root)
    dests = [dest for _src, dest in datas]
    assert not any("sam2" in dest for dest in dests)
