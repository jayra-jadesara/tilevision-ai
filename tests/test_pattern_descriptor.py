"""Tests for pattern structure descriptor (speckled vs marble)."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.descriptors.pattern_descriptor import PatternDescriptor


def _blank_canvas(color_bgr: tuple[int, int, int] = (220, 220, 215)) -> np.ndarray:
    return np.full((256, 256, 3), color_bgr, dtype=np.uint8)


def _make_speckled_image() -> np.ndarray:
    image = _blank_canvas()
    rng = np.random.default_rng(42)
    for _ in range(350):
        x = int(rng.integers(8, 248))
        y = int(rng.integers(8, 248))
        radius = int(rng.integers(1, 3))
        shade = int(rng.integers(40, 120))
        cv2.circle(image, (x, y), radius, (shade, shade, shade), thickness=-1)
    return image


def _make_marble_image() -> np.ndarray:
    image = _blank_canvas((235, 230, 225))
    for offset in range(-80, 320, 18):
        pts = np.array(
            [
                [offset, 0],
                [offset + 90, 256],
                [offset + 110, 256],
                [offset + 20, 0],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(image, [pts], (140, 135, 130))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
    image = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    return image


def test_extract_returns_twelve_dimensional_vector():
    vector = PatternDescriptor.extract(_make_speckled_image())
    assert vector.shape == (PatternDescriptor.FEATURE_SIZE,)


def test_speckled_has_higher_circularity_and_small_blob_ratio_than_marble():
    speckled = PatternDescriptor.extract(_make_speckled_image())
    marble = PatternDescriptor.extract(_make_marble_image())

    assert speckled[7] > marble[7]  # small_blob_ratio
    assert speckled[9] > marble[9]  # mean_circularity
    assert speckled[10] < marble[10]  # vein_coverage
    assert speckled[8] < marble[8]  # elongation_ratio


def test_marble_has_higher_structure_coherence_than_speckled():
    speckled = PatternDescriptor.extract(_make_speckled_image())
    marble = PatternDescriptor.extract(_make_marble_image())

    assert marble[11] > speckled[11]  # structure_coherence


def test_similar_speckled_patterns_score_higher_than_speckled_vs_marble():
    speckled_a = PatternDescriptor.extract(_make_speckled_image())
    speckled_b = PatternDescriptor.extract(_make_speckled_image())
    marble = PatternDescriptor.extract(_make_marble_image())

    same_family = PatternDescriptor.similarity(speckled_a, speckled_b)
    cross_family = PatternDescriptor.similarity(speckled_a, marble)

    assert same_family > cross_family


def test_legacy_eight_dimensional_vectors_still_compare():
    legacy_a = np.array(
        [0.002, 0.0006, 0.0002, 0.35, 0.08, 0.70, 0.55, 0.82],
        dtype=np.float32,
    )
    legacy_b = legacy_a.copy()
    legacy_b[3] = 0.30

    score = PatternDescriptor.similarity(legacy_a, legacy_b)
    assert 0.0 < score < 1.0
