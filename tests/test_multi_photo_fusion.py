"""Mean + coverage fusion for multi-photo FAISS hits."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.search_quality.multi_photo_fusion import (
    collapse_view_lists_to_photo_best,
    fuse_multi_photo_scores,
)


def test_coverage_bonus_prefers_tiles_hit_by_both_photos():
    # Tile 10 matches both moderately; tile 20 matches only A slightly higher.
    # Mean(0.82,0.80)+0.05 coverage beats single-photo 0.86.
    per_photo = [
        {10: 0.82, 20: 0.86},
        {10: 0.80, 30: 0.84},
    ]
    ordered, fused = fuse_multi_photo_scores(per_photo, coverage_bonus=0.05)
    assert ordered[0] == 10
    assert fused[10] > fused[20]
    assert fused[10] > fused[30]


def test_collapse_views_to_photo_best():
    # Photo A: 2 views; photo B: 1 view.
    per_view = [
        ([1, 2], [0.5, 0.9]),
        ([1, 3], [0.8, 0.4]),
        ([2, 1], [0.7, 0.6]),
    ]
    per_photo = collapse_view_lists_to_photo_best(per_view, [2, 1])
    assert len(per_photo) == 2
    assert per_photo[0][1] == 0.8  # max of 0.5 and 0.8
    assert per_photo[0][2] == 0.9
    assert per_photo[1][2] == 0.7
