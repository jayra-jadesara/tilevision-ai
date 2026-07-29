"""Tests for optional SAM2 installer bundling (Transformers + ONNX)."""

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
    monkeypatch.delenv("MACOS_BUILD_ARCH", raising=False)


def test_should_bundle_sam2_default_off():
    assert should_bundle_sam2() is False
    assert should_bundle_sam2_onnx() is False


def test_should_bundle_sam2_auto_includes_mac_intel(monkeypatch):
    """Mac Intel must get ONNX SAM2 — no longer skipped."""
    monkeypatch.setenv("TILEVISION_BUNDLE_SAM2", "auto")
    monkeypatch.setenv("MACOS_BUILD_ARCH", "x64")
    assert should_bundle_sam2() is True
    assert should_bundle_sam2_onnx(macos_arch="x64") is True
    assert should_bundle_sam2_transformers(macos_arch="x64") is False


def test_should_bundle_sam2_auto_windows_and_arm64(monkeypatch):
    monkeypatch.setenv("TILEVISION_BUNDLE_SAM2", "auto")
    assert should_bundle_sam2_onnx() is True
    monkeypatch.setenv("MACOS_BUILD_ARCH", "arm64")
    assert should_bundle_sam2_transformers(macos_arch="arm64") is True
    assert should_bundle_sam2_onnx(macos_arch="arm64") is True


def test_collect_datas_includes_onnx_for_intel(tmp_path, monkeypatch):
    monkeypatch.setenv("TILEVISION_BUNDLE_SAM2", "auto")
    monkeypatch.setenv("MACOS_BUILD_ARCH", "x64")
    root = tmp_path
    (root / "src" / "config").mkdir(parents=True)
    (root / "src" / "config" / "default_config.json").write_text("{}")
    onnx = root / "model_weights" / "sam2.1-hiera-tiny-onnx"
    onnx.mkdir(parents=True)
    (onnx / "sam2.1_hiera_tiny.encoder.onnx").write_bytes(b"x")
    (onnx / "sam2.1_hiera_tiny.decoder.onnx").write_bytes(b"y")
    tr = root / "model_weights" / "sam2.1-hiera-tiny"
    tr.mkdir(parents=True)
    (tr / "config.json").write_text("{}")

    datas = collect_datas(root)
    dests = [dest for _src, dest in datas]
    assert any("sam2.1-hiera-tiny-onnx" in dest for dest in dests)
    assert not any(
        dest.endswith("sam2.1-hiera-tiny") or dest.endswith("model_weights/sam2.1-hiera-tiny")
        for dest in dests
    )


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
