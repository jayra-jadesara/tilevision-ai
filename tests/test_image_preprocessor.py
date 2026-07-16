"""Tests for image preprocessing pipeline."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.preprocess.image_preprocessor import ImagePreprocessor, TARGET_SIZE


def test_letterbox_preserves_aspect_ratio(tmp_path):
    path = tmp_path / "wide.jpg"
    Image.new("RGB", (800, 400), color=(50, 100, 150)).save(path)

    processed = ImagePreprocessor.preprocess(path)

    assert processed.pil.size == (TARGET_SIZE, TARGET_SIZE)
    assert processed.width == 800
    assert processed.height == 400


def test_rgba_composited_on_neutral_background(tmp_path):
    path = tmp_path / "alpha.png"
    img = Image.new("RGBA", (64, 64), color=(255, 0, 0, 128))
    img.save(path)

    processed = ImagePreprocessor.preprocess(path)

    assert processed.pil.mode == "RGB"
    assert processed.rgb.shape == (TARGET_SIZE, TARGET_SIZE, 3)


def test_uniform_white_border_is_trimmed(tmp_path):
    path = tmp_path / "bordered.jpg"
    canvas = np.full((100, 100, 3), 255, dtype=np.uint8)
    canvas[20:80, 20:80] = (40, 80, 120)
    Image.fromarray(canvas).save(path)

    processed = ImagePreprocessor.preprocess(path)

    # Original metadata preserved from source file dimensions.
    assert processed.width == 100
    assert processed.height == 100
    # Tile content should dominate the processed canvas (not only white).
    center = processed.rgb[TARGET_SIZE // 2, TARGET_SIZE // 2]
    assert center.mean() < 240


def test_small_image_still_produces_valid_output(tmp_path):
    path = tmp_path / "tiny.jpg"
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(path)

    processed = ImagePreprocessor.preprocess(path)

    assert processed.pil.size == (TARGET_SIZE, TARGET_SIZE)
    assert processed.bgr.shape == (TARGET_SIZE, TARGET_SIZE, 3)
