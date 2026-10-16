"""
Fuse FAISS hits across multiple query photos.

Within each photo, take MAX cosine across that photo's views (same as
single-photo multi-crop). Across photos, use the mean of per-photo scores
plus a coverage bonus so tiles supported by more than one photo rank above
tiles that only match one.

Choice vs plain MAX across all views: MAX is "similar to A **or** B" and
ranks a one-photo match beside a both-photo match. Mean + coverage rewards
agreement without requiring RRF retuning of the single-photo path.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


DEFAULT_COVERAGE_BONUS = 0.06


def fuse_multi_photo_scores(
    per_photo_best: Sequence[Dict[int, float]],
    *,
    coverage_bonus: float = DEFAULT_COVERAGE_BONUS,
) -> Tuple[List[int], Dict[int, float]]:
    """
    Args:
        per_photo_best: One dict per photo of tile_id → best cosine for that photo.

    Returns:
        Ordered tile ids (desc) and fused scores (used as cosine for display).
    """
    if not per_photo_best:
        return [], {}

    all_ids: set[int] = set()
    for d in per_photo_best:
        all_ids.update(int(k) for k in d.keys())

    fused: Dict[int, float] = {}
    for tid in all_ids:
        scores = [float(d[tid]) for d in per_photo_best if tid in d]
        if not scores:
            continue
        mean = sum(scores) / len(scores)
        coverage = max(0, len(scores) - 1)
        fused[tid] = mean + coverage_bonus * coverage

    ordered = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)
    return ordered, fused


def collapse_view_lists_to_photo_best(
    per_view_lists: Sequence[Tuple[Sequence[int], Sequence[float]]],
    photo_view_counts: Sequence[int],
) -> List[Dict[int, float]]:
    """
    Group flat per-view FAISS lists into per-photo best-score dicts.

    ``photo_view_counts`` sums to ``len(per_view_lists)``.
    """
    if sum(int(c) for c in photo_view_counts) != len(per_view_lists):
        raise ValueError(
            f"photo_view_counts {list(photo_view_counts)} do not match "
            f"{len(per_view_lists)} view lists"
        )
    out: List[Dict[int, float]] = []
    offset = 0
    for count in photo_view_counts:
        n = int(count)
        best: Dict[int, float] = {}
        for ids, scores in per_view_lists[offset : offset + n]:
            for tid, sc in zip(ids, scores):
                tid_i = int(tid)
                sc_f = float(sc)
                prev = best.get(tid_i)
                if prev is None or sc_f > prev:
                    best[tid_i] = sc_f
        out.append(best)
        offset += n
    return out
