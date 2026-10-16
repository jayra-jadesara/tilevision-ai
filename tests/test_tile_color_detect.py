from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.tile_color_detect import detect_color_from_image, resolve_index_color


def _solid(path: Path, rgb) -> Path:
    Image.new("RGB", (64, 64), color=rgb).save(path)
    return path


def test_detect_white_and_gray(tmp_path):
    white = _solid(tmp_path / "w.png", (245, 245, 245))
    gray = _solid(tmp_path / "g.png", (130, 130, 130))
    assert detect_color_from_image(white) == "White"
    assert detect_color_from_image(gray) == "Gray"


def test_filename_wins_over_pixels(tmp_path):
    path = _solid(tmp_path / "x.png", (20, 20, 20))
    color, source = resolve_index_color("Beige", path)
    assert color == "Beige"
    assert source == "filename"


def test_auto_from_pixels_when_unknown(tmp_path):
    path = _solid(tmp_path / "x.png", (20, 20, 20))
    color, source = resolve_index_color("Unknown", path)
    assert color == "Black"
    assert source == "auto"
