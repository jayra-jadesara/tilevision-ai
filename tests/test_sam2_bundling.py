"""Tests for optional SAM2 installer bundling."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packaging"))

from pyinstaller_common import (  # noqa: E402
    collect_datas,
    should_bundle_sam2,
)


@pytest.fixture(autouse=True)
def _clear_bundle_flag(monkeypatch):
    monkeypatch.delenv("TILEVISION_BUNDLE_SAM2", raising=False)
    monkeypatch.delenv("MACOS_BUILD_ARCH", raising=False)


def test_should_bundle_sam2_default_off():
    assert should_bundle_sam2() is False


def test_should_bundle_sam2_on(monkeypatch):
    monkeypatch.setenv("TILEVISION_BUNDLE_SAM2", "1")
    assert should_bundle_sam2() is True


def test_should_bundle_sam2_auto_skips_mac_intel(monkeypatch):
    monkeypatch.setenv("TILEVISION_BUNDLE_SAM2", "auto")
    monkeypatch.setenv("MACOS_BUILD_ARCH", "x64")
    assert should_bundle_sam2() is False
    assert should_bundle_sam2(macos_arch="x64") is False


def test_should_bundle_sam2_auto_allows_arm64_and_windows(monkeypatch):
    monkeypatch.setenv("TILEVISION_BUNDLE_SAM2", "auto")
    monkeypatch.setenv("MACOS_BUILD_ARCH", "arm64")
    assert should_bundle_sam2() is True
    monkeypatch.delenv("MACOS_BUILD_ARCH", raising=False)
    # No Mac arch → treat as Windows/Linux build host.
    assert should_bundle_sam2() is True


def test_collect_datas_omits_sam2_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("TILEVISION_BUNDLE_SAM2", raising=False)
    root = tmp_path
    (root / "src" / "config").mkdir(parents=True)
    (root / "src" / "config" / "default_config.json").write_text("{}")
    sam2 = root / "model_weights" / "sam2.1-hiera-tiny"
    sam2.mkdir(parents=True)
    (sam2 / "config.json").write_text("{}")

    datas = collect_datas(root)
    dests = [dest for _src, dest in datas]
    assert not any("sam2.1-hiera-tiny" in dest for dest in dests)


def test_collect_datas_includes_sam2_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("TILEVISION_BUNDLE_SAM2", "1")
    root = tmp_path
    (root / "src" / "config").mkdir(parents=True)
    (root / "src" / "config" / "default_config.json").write_text("{}")
    sam2 = root / "model_weights" / "sam2.1-hiera-tiny"
    sam2.mkdir(parents=True)
    (sam2 / "config.json").write_text("{}")

    datas = collect_datas(root)
    dests = [dest for _src, dest in datas]
    assert any(dest.endswith("sam2.1-hiera-tiny") or "sam2.1-hiera-tiny" in dest for dest in dests)
