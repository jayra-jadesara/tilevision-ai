"""Tests for cross-platform image format helpers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.utils.image_formats as image_formats


def test_base_extensions_without_heif(monkeypatch):
    monkeypatch.setattr(image_formats, "_heif_registered", False)
    exts = image_formats.supported_image_extensions()
    assert ".jpg" in exts
    assert ".heic" not in exts


def test_heic_when_pillow_heif_registered(monkeypatch):
    monkeypatch.setattr(image_formats, "_heif_registered", True)
    exts = image_formats.query_image_extensions()
    assert ".heic" in exts
    assert ".heif" in exts
