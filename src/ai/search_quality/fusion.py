"""
Tile-id score fusion strategies for multi-vector FAISS hits.

Weights are tuned on the golden validation set — never hard-coded by gut.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import numpy as np


class FusionMethod(str, Enum):
    MAX = "max"
    WEIGHTED_MAX = "weighted_max"
    AVERAGE = "average"
    WEIGHTED_AVERAGE = "weighted_average"
    RRF = "rrf"
    SOFTMAX = "softmax"


@dataclass(frozen=True, slots=True)
class ScoredHit:
    tile_id: int
    score: float
    view_weight: float = 1.0
    rank_in_list: int = 1  # 1-based within a raw FAISS list


def fuse_hits(
    hits: Sequence[ScoredHit],
    method: FusionMethod | str,
    *,
    rrf_k: int = 60,
    view_weights: dict[str, float] | None = None,
) -> list[tuple[int, float]]:
    """
    Collapse per-vector hits to one score per tile_id.

    Returns (tile_id, fused_score) sorted descending.
    """
    if not isinstance(method, FusionMethod):
        method = FusionMethod(method)

    if method == FusionMethod.MAX:
        best: dict[int, float] = {}
        for h in hits:
            prev = best.get(h.tile_id)
            if prev is None or h.score > prev:
                best[h.tile_id] = h.score
        return sorted(best.items(), key=lambda x: x[1], reverse=True)

    if method == FusionMethod.WEIGHTED_MAX:
        best = {}
        for h in hits:
            w = max(0.05, float(h.view_weight))
            val = h.score * w
            prev = best.get(h.tile_id)
            if prev is None or val > prev:
                best[h.tile_id] = val
        return sorted(best.items(), key=lambda x: x[1], reverse=True)

    if method == FusionMethod.AVERAGE:
        sums: dict[int, float] = defaultdict(float)
        counts: dict[int, int] = defaultdict(int)
        for h in hits:
            sums[h.tile_id] += h.score
            counts[h.tile_id] += 1
        fused = {tid: sums[tid] / max(1, counts[tid]) for tid in sums}
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    if method == FusionMethod.WEIGHTED_AVERAGE:
        sums = defaultdict(float)
        weights = defaultdict(float)
        for h in hits:
            w = max(0.05, float(h.view_weight))
            sums[h.tile_id] += h.score * w
            weights[h.tile_id] += w
        fused = {tid: sums[tid] / max(1e-8, weights[tid]) for tid in sums}
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    if method == FusionMethod.RRF:
        scores: dict[int, float] = defaultdict(float)
        for h in hits:
            scores[h.tile_id] += 1.0 / (rrf_k + max(1, int(h.rank_in_list)))
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if method == FusionMethod.SOFTMAX:
        # Softmax over raw scores within each tile_id, then sum mass.
        by_tile: dict[int, list[float]] = defaultdict(list)
        for h in hits:
            by_tile[h.tile_id].append(float(h.score) * max(0.05, float(h.view_weight)))
        fused: dict[int, float] = {}
        for tid, vals in by_tile.items():
            arr = np.asarray(vals, dtype=np.float64)
            arr = arr - arr.max()
            ex = np.exp(arr)
            fused[tid] = float(ex.sum())  # total probability mass proxy
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    raise ValueError(f"Unknown fusion method: {method}")


def tune_weighted_max(
    trials: Iterable[tuple[list[ScoredHit], int]],
    weight_grid: Sequence[float] = (0.70, 0.80, 0.90, 1.0, 1.05, 1.10),
) -> tuple[float, float]:
    """
    Grid-search a global aux view weight for WEIGHTED_MAX.

    Each trial is (hits with view_weight already set for primary=1.0 / aux=?),
    relevant_tile_id). Returns (best_aux_weight, recall_at_1).
    """
    best_w = 1.0
    best_r1 = -1.0
    trials_list = list(trials)
    if not trials_list:
        return 1.0, 0.0

    for w in weight_grid:
        hits_r1 = 0
        for hits, relevant in trials_list:
            adjusted = [
                ScoredHit(
                    tile_id=h.tile_id,
                    score=h.score,
                    view_weight=1.0 if h.view_weight >= 0.999 else w,
                    rank_in_list=h.rank_in_list,
                )
                for h in hits
            ]
            fused = fuse_hits(adjusted, FusionMethod.WEIGHTED_MAX)
            if fused and fused[0][0] == relevant:
                hits_r1 += 1
        r1 = hits_r1 / len(trials_list)
        if r1 > best_r1:
            best_r1 = r1
            best_w = float(w)
    return best_w, float(best_r1)
