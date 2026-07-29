"""Tests for enable_sam2_precise_crop setting."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import AppSettings


def test_enable_sam2_precise_crop_defaults_on(tmp_path):
    settings = AppSettings(config_dir=tmp_path / "cfg")
    assert settings.enable_sam2_precise_crop is True


def test_enable_sam2_precise_crop_persists_off(tmp_path):
    settings = AppSettings(config_dir=tmp_path / "cfg")
    settings.enable_sam2_precise_crop = False
    reloaded = AppSettings(config_dir=tmp_path / "cfg")
    assert reloaded.enable_sam2_precise_crop is False
