"""Tests for ORB local-feature geometric verification."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.verification.orb_verifier import OrbVerifier
from src.config.settings import AppSettings


def _textured_gray(seed: int = 0, size: int = 256) -> np.ndarray:
    """Synthetic marble-like grayscale with enough structure for ORB."""
    rng = np.random.default_rng(seed)
    base = np.full((size, size), 200, dtype=np.uint8)
    for _ in range(80):
        x0, y0 = int(rng.integers(0, size)), int(rng.integers(0, size))
        x1, y1 = int(rng.integers(0, size)), int(rng.integers(0, size))
        c = int(rng.integers(40, 180))
        cv2.line(base, (x0, y0), (x1, y1), c, 1 + (seed % 3), cv2.LINE_AA)
    for _ in range(12):
        cx, cy = int(rng.integers(0, size)), int(rng.integers(0, size))
        rad = int(rng.integers(8, 40))
        shade = int(rng.integers(60, 190))
        overlay = base.copy()
        cv2.circle(overlay, (cx, cy), rad, shade, -1)
        base = cv2.addWeighted(overlay, 0.35, base, 0.65, 0)
    return base


def test_orb_same_image_scores_high():
    gray = _textured_gray(seed=7)
    verifier = OrbVerifier()
    score = verifier.score(gray, gray.copy())
    assert score >= 0.7


def test_orb_rotated_crop_still_meaningful():
    gray = _textured_gray(seed=11)
    h, w = gray.shape
    # Center crop then slight rotation — near-duplicate under transform.
    crop = gray[h // 8 : h - h // 8, w // 8 : w - w // 8]
    M = cv2.getRotationMatrix2D((crop.shape[1] / 2, crop.shape[0] / 2), 8, 1.0)
    rotated = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]))
    verifier = OrbVerifier()
    score = verifier.score(gray, rotated)
    assert score > 0.15


def test_orb_different_images_score_low():
    a = _textured_gray(seed=1)
    b = _textured_gray(seed=99)
    # Force very different structure: noise field.
    rng = np.random.default_rng(123)
    b = rng.integers(0, 255, size=a.shape, dtype=np.uint8)
    verifier = OrbVerifier()
    score = verifier.score(a, b)
    assert score < 0.35


def test_orb_blank_image_returns_zero():
    blank = np.full((128, 128), 255, dtype=np.uint8)
    textured = _textured_gray(seed=3)
    verifier = OrbVerifier()
    assert verifier.score(blank, textured) == 0.0
    assert verifier.score(textured, blank) == 0.0


def test_orb_tiny_image_returns_zero():
    tiny = np.zeros((8, 8), dtype=np.uint8)
    textured = _textured_gray(seed=4)
    verifier = OrbVerifier()
    assert verifier.score(tiny, textured) == 0.0


def test_orb_never_raises_on_malformed():
    verifier = OrbVerifier()
    assert verifier.score(np.array([]), _textured_gray()) == 0.0
    assert verifier.score(None, _textured_gray()) == 0.0  # type: ignore[arg-type]
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    assert 0.0 <= verifier.score(rgb, rgb) <= 1.0


def test_enable_orb_verification_defaults_off(tmp_path):
    settings = AppSettings(config_dir=tmp_path / "cfg")
    assert settings.enable_orb_verification is False


def test_enable_orb_verification_persists_off(tmp_path):
    settings = AppSettings(config_dir=tmp_path / "cfg")
    settings.enable_orb_verification = False
    reloaded = AppSettings(config_dir=tmp_path / "cfg")
    assert reloaded.enable_orb_verification is False
