"""Rebuild ETA must ignore warmup batches and track recent steady-state rate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.use_cases.index_eta import typical_work_seconds
from src.utils.duration_format import format_eta_seconds


def test_warmup_batch_does_not_inflate_typical():
    # First batch after cold model load is often many× slower.
    batches = [8.0, 1.0, 1.1, 0.9, 1.0]
    mean_old = sum(batches) / len(batches)
    typical = typical_work_seconds(batches)
    assert mean_old > 2.0
    assert typical < 1.2


def test_warmup_alone_returns_zero_until_second_batch():
    """
    Real Windows log: first batch per_image=59.077s (model load) vs steady
    ~12.6 s. Using the warm-up alone projected "~46 min" when true total
    was ~14 min. ETA must stay indeterminate (0 → UI 'estimating…').
    """
    assert typical_work_seconds([59.077]) == 0.0
    assert typical_work_seconds([29.556]) == 0.0
    # After a second (steady) sample, use post-warmup only.
    typical = typical_work_seconds([59.077, 12.6])
    assert abs(typical - 12.6) < 0.01
    remaining = 34  # ~38 - first batch of 4
    eta = typical * remaining
    # Old (warmup-fed) ETA would be ~2000s (~33 min); new is ~7 min class.
    assert eta < 500
    assert typical_work_seconds([59.077]) * remaining == 0.0


def test_eta_at_60_percent_matches_user_304s_scenario():
    """
    Reproduced user report shape: early batches ~12–45s/image while models
    warm → old mean projected ~315s remaining at 60%. Recent-batch median
    (skip first) tracks the improving rate instead.
    """
    total = 40
    batch_size = 4
    batches = [45.0, 25.0, 20.0, 18.0, 15.0, 12.0]  # 24 files = 60%
    work = []
    for pi in batches:
        work.extend([pi] * batch_size)
    remaining = total - 24
    old_eta = (sum(work) / len(work)) * remaining
    new_eta = typical_work_seconds(batches) * remaining
    assert old_eta > 280
    assert new_eta < old_eta
    assert new_eta < 280


def test_eta_tracks_speedup_after_warmup():
    batches = [20.0, 1.0, 1.0, 0.9, 1.1, 1.0]
    remaining = 16
    old_expanded = []
    for pi in batches:
        old_expanded.extend([pi] * 4)
    old_eta = (sum(old_expanded) / len(old_expanded)) * remaining
    new_eta = typical_work_seconds(batches) * remaining
    assert old_eta > new_eta * 2
    assert abs(new_eta - 16.0) < 2.0  # ~1s/file * 16


def test_format_eta_human_readable():
    assert format_eta_seconds(40) == "~40 s"
    assert format_eta_seconds(14 * 60) == "~14 min"
    assert "h" in format_eta_seconds(11500)
    assert format_eta_seconds(3 * 3600 + 12 * 60) == "~3 h 12 min"
    assert format_eta_seconds(0) == ""
    assert format_eta_seconds(-1) == ""
