"""Tests for LAB color descriptor."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.descriptors.color_descriptor import ColorDescriptor


def _solid_bgr_image(color_bgr: tuple[int, int, int]) -> np.ndarray:
    rgb = Image.new("RGB", (64, 64), color=(color_bgr[2], color_bgr[1], color_bgr[0]))
    return np.asarray(rgb)[:, :, ::-1].copy()


def test_extract_returns_expected_vector_size():
    image = _solid_bgr_image((40, 80, 200))
    vector = ColorDescriptor.extract(image)
    assert vector.shape == (ColorDescriptor.vector_size(),)


def test_similar_tiles_score_higher_than_different_colors():
    white = ColorDescriptor.extract(_solid_bgr_image((240, 240, 240)))
    off_white = ColorDescriptor.extract(_solid_bgr_image((220, 225, 235)))
    dark_blue = ColorDescriptor.extract(_solid_bgr_image((180, 40, 20)))

    similar = ColorDescriptor.similarity(white, off_white)
    different = ColorDescriptor.similarity(white, dark_blue)

    assert similar > different


def test_lab_distance_is_smaller_for_similar_colors():
    white = (240, 240, 240)
    cream = (230, 225, 210)
    navy = (20, 20, 120)

    assert ColorDescriptor.rgb_to_lab_distance(white, cream) < (
        ColorDescriptor.rgb_to_lab_distance(white, navy)
    )
