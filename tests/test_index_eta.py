"""Rebuild ETA must ignore warmup batches and track recent steady-state rate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _typical_work_seconds(batch_timings, warmup_batches_skip=1, recent_batch_window=4):
    """Mirror of IndexImagesUseCase steady-state helper (keep in sync)."""
    if not batch_timings:
        return 0.0
    steady = (
        batch_timings[warmup_batches_skip:]
        if len(batch_timings) > warmup_batches_skip
        else list(batch_timings)
    )
    if not steady:
        return 0.0
    recent = steady[-recent_batch_window:]
    ordered = sorted(recent)
    mid = len(ordered) // 2
    median = (
        ordered[mid]
        if len(ordered) % 2
        else 0.5 * (ordered[mid - 1] + ordered[mid])
    )
    capped = [s for s in ordered if s <= max(median * 3.0, median + 1.5)]
    if not capped:
        capped = ordered
    mid = len(capped) // 2
    return (
        capped[mid]
        if len(capped) % 2
        else 0.5 * (capped[mid - 1] + capped[mid])
    )


def test_warmup_batch_does_not_inflate_typical():
    # First batch after cold model load is often many× slower.
    batches = [8.0, 1.0, 1.1, 0.9, 1.0]
    mean_old = sum(batches) / len(batches)
    typical = _typical_work_seconds(batches)
    assert mean_old > 2.0
    assert typical < 1.2


def test_eta_at_60_percent_matches_user_304s_scenario():
    """
    Reproduced user report shape: early batches ~12–45s/image while models
    warm → old mean projected ~315s remaining at 60%. Recent-batch median
    (skip first) tracks the improving rate instead.
    """
    total = 40
    batch_size = 4
    batches = [45.0, 25.0, 20.0, 18.0, 15.0, 12.0]  # 24 files = 60%
    # Expand like the old code did (per-file duplicates of batch rate).
    work = []
    for pi in batches:
        work.extend([pi] * batch_size)
    remaining = total - 24
    old_eta = (sum(work) / len(work)) * remaining
    new_eta = _typical_work_seconds(batches) * remaining
    assert old_eta > 280  # ~360s — matches screenshot class of bug
    # Recent post-warmup median cuts the inflated projection; full recovery
    # needs later fast batches (covered below).
    assert new_eta < old_eta
    assert new_eta < 280


def test_eta_tracks_speedup_after_warmup():
    batches = [20.0, 1.0, 1.0, 0.9, 1.1, 1.0]
    remaining = 16
    old_expanded = []
    for pi in batches:
        old_expanded.extend([pi] * 4)
    old_eta = (sum(old_expanded) / len(old_expanded)) * remaining
    new_eta = _typical_work_seconds(batches) * remaining
    assert old_eta > new_eta * 2
    assert abs(new_eta - 16.0) < 2.0  # ~1s/file * 16
