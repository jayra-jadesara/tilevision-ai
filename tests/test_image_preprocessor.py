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


def test_scene_photo_detection_on_wide_image(tmp_path):
    path = tmp_path / "room.jpg"
    Image.new("RGB", (800, 400), color=(200, 200, 200)).save(path)
    with Image.open(path) as img:
        assert ImagePreprocessor._looks_like_scene_photo(img.convert("RGB"))


def test_preprocess_for_query_runs_without_error(tmp_path):
    path = tmp_path / "tile.jpg"
    Image.new("RGB", (256, 256), color=(180, 180, 180)).save(path)
    processed = ImagePreprocessor.preprocess_for_query(path)
    assert processed.pil.size == (TARGET_SIZE, TARGET_SIZE)


def test_small_image_still_produces_valid_output(tmp_path):
    path = tmp_path / "tiny.jpg"
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(path)

    processed = ImagePreprocessor.preprocess(path)

    assert processed.pil.size == (TARGET_SIZE, TARGET_SIZE)
    assert processed.bgr.shape == (TARGET_SIZE, TARGET_SIZE, 3)


def test_normalize_lighting_skips_high_key_low_contrast_marble():
    """
    Cream marble has a naturally narrow L-range. Stretching it posterizes
    veins (PGYS2319 panel primary). Must leave the image essentially unchanged.
    """
    rng = np.random.default_rng(7)
    # Soft high-key marble: mean ~230, span of veins ~20–30.
    base = np.full((256, 256, 3), 235, dtype=np.float32)
    for _ in range(40):
        x0, y0 = int(rng.integers(0, 256)), int(rng.integers(0, 256))
        x1, y1 = int(rng.integers(0, 256)), int(rng.integers(0, 256))
        color = float(rng.integers(210, 235))
        rr = np.linspace(y0, y1, 50).astype(int).clip(0, 255)
        cc = np.linspace(x0, x1, 50).astype(int).clip(0, 255)
        base[rr, cc] = color
    marble = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))

    # Confirm the old gate would have fired (span < 40).
    import cv2

    lab = cv2.cvtColor(np.asarray(marble)[:, :, ::-1], cv2.COLOR_BGR2LAB)
    low, high = np.percentile(lab[:, :, 0], (2, 98))
    assert high - low < 40, f"fixture span too wide: {high - low}"

    out = ImagePreprocessor.normalize_lighting(marble)
    delta = float(np.mean(np.abs(np.asarray(out, dtype=np.float32) - np.asarray(marble))))
    assert delta < 2.0, f"high-key marble was contrast-stretched (mean abs Δ={delta:.2f})"


def test_normalize_lighting_still_corrects_underexposed_frame():
    """Dark crushed frames must still receive the L-channel stretch."""
    rng = np.random.default_rng(3)
    # Underexposed: mean ~70, narrow span ~25.
    base = np.full((256, 256, 3), 65, dtype=np.float32)
    base += rng.normal(0, 8, base.shape)
    dark = Image.fromarray(np.clip(base, 40, 95).astype(np.uint8))

    out = ImagePreprocessor.normalize_lighting(dark)
    in_mean = float(np.asarray(dark).mean())
    out_mean = float(np.asarray(out).mean())
    assert out_mean > in_mean + 20, (
        f"underexposed frame not brightened: {in_mean:.1f} → {out_mean:.1f}"
    )
    # Output should use more of the 0–255 range.
    out_l = np.asarray(out.convert("L"), dtype=np.float32)
    assert float(out_l.max() - out_l.min()) > 100


def test_load_downscales_huge_images_before_processing(tmp_path):
    path = tmp_path / "huge.jpg"
    Image.new("RGB", (6000, 4000), color=(90, 120, 150)).save(path, quality=95)

    ImagePreprocessor.configure(max_decode_edge=1024)
    try:
        with Image.open(path) as original:
            original = original.convert("RGB")
            loaded = ImagePreprocessor.load(path)
            assert max(loaded.size) <= 1024
            assert loaded.size != original.size
    finally:
        ImagePreprocessor.configure(max_decode_edge=2048)
