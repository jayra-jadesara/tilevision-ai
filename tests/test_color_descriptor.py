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


def test_near_white_marble_survives_white_balance_shift():
    """
    Same marble under cool vs warm WB used to score ~0.001–0.18 on LAB hist
    CORREL. Near-white soft path must keep them clearly similar.
    """
    rng = np.random.default_rng(7)
    base = np.full((128, 128, 3), 230, dtype=np.float32)
    for _ in range(30):
        x0, y0 = int(rng.integers(0, 128)), int(rng.integers(0, 128))
        x1, y1 = int(rng.integers(0, 128)), int(rng.integers(0, 128))
        color = int(rng.integers(200, 235))
        # draw via numpy slice approximation
        rr = np.linspace(y0, y1, 40).astype(int).clip(0, 127)
        cc = np.linspace(x0, x1, 40).astype(int).clip(0, 127)
        base[rr, cc] = color

    cool = np.clip(base * [0.98, 0.99, 1.05], 0, 255).astype(np.uint8)
    warm = np.clip(base * [1.05, 1.01, 0.96], 0, 255).astype(np.uint8)
    # ColorDescriptor expects BGR
    cool_bgr = cool[:, :, ::-1].copy()
    warm_bgr = warm[:, :, ::-1].copy()

    cool_v = ColorDescriptor.extract(cool_bgr)
    warm_v = ColorDescriptor.extract(warm_bgr)
    score = ColorDescriptor.similarity(cool_v, warm_v)
    assert score >= 0.55, f"near-white WB pair scored too low: {score:.3f}"

    # Still discriminate white marble from saturated blue.
    blue = ColorDescriptor.extract(_solid_bgr_image((180, 40, 20)))
    assert ColorDescriptor.similarity(cool_v, blue) < score


def test_lab_distance_is_smaller_for_similar_colors():
    white = (240, 240, 240)
    cream = (230, 225, 210)
    navy = (20, 20, 120)

    assert ColorDescriptor.rgb_to_lab_distance(white, cream) < (
        ColorDescriptor.rgb_to_lab_distance(white, navy)
    )


def test_is_near_white_rgb_gates():
    assert ColorDescriptor.is_near_white_rgb((245, 245, 245))
    assert ColorDescriptor.is_near_white_rgb((230, 225, 210))
    assert not ColorDescriptor.is_near_white_rgb((20, 20, 120))
    assert not ColorDescriptor.is_near_white_rgb((200, 40, 40))
