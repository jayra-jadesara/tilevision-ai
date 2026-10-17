from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.search_quality.description_colour_fusion import (
    extract_colour_tokens,
    fuse_clip_with_colour_rrf,
)


def test_extract_colour_tokens_british_and_american():
    assert "gray" in extract_colour_tokens("grey stone floor")
    assert "white" in extract_colour_tokens("matte white subway")


def test_non_colour_query_unchanged_order():
    pairs = [(1, 0.9), (2, 0.8), (3, 0.7)]
    out = fuse_clip_with_colour_rrf(pairs, {1: "Gray", 2: "White"}, [])
    assert out == pairs


def test_colour_query_promotes_matching_label():
    pairs = [(1, 0.31), (2, 0.30), (3, 0.29)]
    colors = {1: "White", 2: "Gray", 3: "Gray"}
    out = fuse_clip_with_colour_rrf(pairs, colors, ["gray"])
    assert out[0][0] in (2, 3)
    assert out[0][0] != 1
