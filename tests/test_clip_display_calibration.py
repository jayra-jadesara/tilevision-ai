"""CLIP display % calibration must not reuse the DINOv2 hybrid curve."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.similarity_score import (
    calibrate_clip_display_percent,
    calibrate_display_percent,
)


def test_clip_calibration_lifts_typical_good_cosine():
    # Production "gray tile" observation: ~17.8% under DINOv2 curve ⇒ raw ≈ 0.27
    raw = 0.27
    dino = calibrate_display_percent(raw)
    clip = calibrate_clip_display_percent(raw)
    assert dino < 20.0
    assert clip >= 75.0
    assert clip <= 90.0


def test_clip_calibration_keeps_weak_cosine_low():
    # Unrelated text (e.g. "sushi") cluster ~0.16 on measured catalog
    raw = 0.16
    clip = calibrate_clip_display_percent(raw)
    assert clip < 30.0
    assert clip > 10.0


def test_clip_calibration_monotonic():
    scores = [0.12, 0.18, 0.22, 0.25, 0.28, 0.35]
    mapped = [calibrate_clip_display_percent(s) for s in scores]
    assert mapped == sorted(mapped)
    assert mapped[-1] < 100.0


def test_measured_gray_tile_cluster_not_low_confidence_display():
    # Synthetic catalog top hits for "gray tile" were ~0.23 cosine
    clip = calibrate_clip_display_percent(0.233)
    assert clip >= 55.0
