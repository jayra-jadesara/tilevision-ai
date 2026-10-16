"""Steady-state seconds/image for rebuild ETA (shared with unit tests)."""

from __future__ import annotations

from typing import List, Sequence


def typical_work_seconds(
    batch_timings: Sequence[float],
    *,
    warmup_batches_skip: int = 1,
    recent_batch_window: int = 4,
) -> float:
    """
    Median seconds/image from recent *post-warmup* batches.

    The first flush usually includes one-time DINOv2/CLIP model load. Using
    that alone inflated ETA 3–5× on real Windows CPU logs (59 s/image warmup
    vs ~12.6 s steady). Return 0.0 until a post-warmup sample exists so the
    UI shows "estimating…" instead of a fabricated remaining time.
    """
    if not batch_timings:
        return 0.0
    if len(batch_timings) <= warmup_batches_skip:
        return 0.0
    steady: List[float] = list(batch_timings[warmup_batches_skip:])
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
