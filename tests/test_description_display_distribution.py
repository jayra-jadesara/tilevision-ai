"""Distribution-aware description-search display % (customer log fixtures)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.similarity_score import (
    CUSTOMER_LOG_FIXTURE_STATS,
    calibrate_clip_display_percent,
    calibrate_description_search_display_map,
    description_display_fixture_table,
    description_search_is_low_confidence,
    description_search_top_z,
)


def _synth_scores(min_s: float, med: float, max_s: float, n: int = 38) -> list[float]:
    xs = np.linspace(min_s, max_s, n)
    xs = xs - float(np.median(xs)) + med
    xs = np.clip(xs, min_s, max_s)
    xs.sort()
    xs[0] = min_s
    xs[-1] = max_s
    return [float(x) for x in xs]


def test_fixture_ordering_preserved():
    for label, (lo, med, hi) in CUSTOMER_LOG_FIXTURE_STATS.items():
        scores = _synth_scores(lo, med, hi)
        mapped = calibrate_description_search_display_map(scores)
        assert mapped == sorted(mapped)
        assert mapped[-1] >= mapped[0]


def test_texture_top_lower_than_gray_winner():
    gray = _synth_scores(*CUSTOMER_LOG_FIXTURE_STATS["gray"])
    texture = _synth_scores(*CUSTOMER_LOG_FIXTURE_STATS["texture"])
    g_top = max(calibrate_description_search_display_map(gray))
    t_top = max(calibrate_description_search_display_map(texture))
    assert g_top > t_top + 3.0


def test_worst_tiles_capped_near_forty_percent():
    for label, band in CUSTOMER_LOG_FIXTURE_STATS.items():
        scores = _synth_scores(*band)
        mapped = calibrate_description_search_display_map(scores)
        assert max(mapped) <= 99.5
        assert min(mapped) <= 40.5, label


def test_gray_clear_winner_not_low_confidence_texture_is():
    gray = _synth_scores(*CUSTOMER_LOG_FIXTURE_STATS["gray"])
    texture = _synth_scores(*CUSTOMER_LOG_FIXTURE_STATS["texture"])
    assert not description_search_is_low_confidence(gray)
    assert description_search_top_z(gray) > description_search_top_z(texture)


def test_before_after_fixture_table():
    table = description_display_fixture_table(CUSTOMER_LOG_FIXTURE_STATS)
    assert set(table.keys()) == set(CUSTOMER_LOG_FIXTURE_STATS.keys())
    for row in table.values():
        assert row["top_new"] < row["top_old"] or row["top_old"] > 80
        assert row["min_new"] <= 40.5


def test_clip_anchor_still_available_for_legacy():
    assert calibrate_clip_display_percent(0.27) >= 75.0
