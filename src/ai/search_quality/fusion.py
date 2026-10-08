"""
Tile-id score fusion strategies for multi-vector FAISS hits.

Weights are tuned on the golden validation set — never hard-coded by gut.

Production multi-crop FAISS merge (``SearchTilesUseCase``) historically used
MAX across query views. Reciprocal Rank Fusion (RRF) is available behind
``multi_crop_fusion`` / ``TILEVISION_MULTI_CROP_FUSION`` (default: max).
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger("tilevision.ai.search_quality.fusion")

# Cormack et al. default; bakeoff-tuned candidates live in compare scripts.
DEFAULT_RRF_K = 60
_FUSION_ENV = "TILEVISION_MULTI_CROP_FUSION"
_RRF_K_ENV = "TILEVISION_MULTI_CROP_RRF_K"


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


@dataclass(frozen=True, slots=True)
class MultiViewMergeResult:
    """Ordered tile ids plus MAX cosine (for aux-boost / ORB banding)."""

    matching_ids: list[int]
    best_cosine: dict[int, float]
    best_view: dict[int, int]
    method: FusionMethod
    rrf_k: int


def resolve_multi_crop_fusion(
    configured: str | None = None,
) -> tuple[FusionMethod, int]:
    """
    Resolve production multi-crop FAISS merge method.

    Precedence: ``TILEVISION_MULTI_CROP_FUSION`` env → ``configured`` → max.
    Only ``max`` and ``rrf`` are accepted for production (other bakeoff methods
    stay in ``fuse_hits`` for offline studies).
    """
    raw = (os.environ.get(_FUSION_ENV) or configured or FusionMethod.MAX.value)
    raw = str(raw).strip().lower()
    if raw in {"rrf", "reciprocal_rank", "reciprocal_rank_fusion"}:
        method = FusionMethod.RRF
    else:
        if raw not in {"max", "maximum", FusionMethod.MAX.value}:
            logger.warning(
                "Unknown multi_crop_fusion=%r — using max",
                raw,
            )
        method = FusionMethod.MAX

    k_raw = os.environ.get(_RRF_K_ENV, "").strip()
    rrf_k = DEFAULT_RRF_K
    if k_raw:
        try:
            rrf_k = max(1, int(k_raw))
        except ValueError:
            logger.warning(
                "Ignoring invalid %s=%r — using k=%d",
                _RRF_K_ENV,
                k_raw,
                DEFAULT_RRF_K,
            )
    return method, rrf_k


def hits_from_faiss_lists(
    per_view_lists: Sequence[tuple[Sequence[int], Sequence[float]]],
    *,
    view_weight: float = 1.0,
) -> tuple[list[ScoredHit], dict[int, float], dict[int, int]]:
    """
    Build per-view ranked hits for fusion.

    Within each view, collapse duplicate tile ids to the best cosine, then
    assign 1-based ranks. Also track global MAX cosine + winning view index
    (production aux-boost / logging still need true cosine, not RRF mass).
    """
    hits: list[ScoredHit] = []
    best_cosine: dict[int, float] = {}
    best_view: dict[int, int] = {}

    for view_idx, (ids, scores) in enumerate(per_view_lists):
        view_best: dict[int, float] = {}
        for tile_id, score in zip(ids, scores):
            tid = int(tile_id)
            sc = float(score)
            prev = view_best.get(tid)
            if prev is None or sc > prev:
                view_best[tid] = sc
            gprev = best_cosine.get(tid)
            if gprev is None or sc > gprev:
                best_cosine[tid] = sc
                best_view[tid] = view_idx

        ranked = sorted(view_best.items(), key=lambda item: item[1], reverse=True)
        for rank, (tid, sc) in enumerate(ranked, start=1):
            hits.append(
                ScoredHit(
                    tile_id=tid,
                    score=sc,
                    view_weight=view_weight,
                    rank_in_list=rank,
                )
            )
    return hits, best_cosine, best_view


def merge_multi_view_faiss(
    per_view_lists: Sequence[tuple[Sequence[int], Sequence[float]]],
    method: FusionMethod | str = FusionMethod.MAX,
    *,
    rrf_k: int = DEFAULT_RRF_K,
    view_weight: float = 1.0,
) -> MultiViewMergeResult:
    """
    Merge per-query-view FAISS lists.

    Ordering follows ``method``. ``best_cosine`` is always MAX cosine so
    downstream aux-boost / ORB banding stay on the similarity scale.
    """
    if not isinstance(method, FusionMethod):
        method = FusionMethod(method)
    hits, best_cosine, best_view = hits_from_faiss_lists(
        per_view_lists, view_weight=view_weight
    )
    if not hits:
        return MultiViewMergeResult([], {}, {}, method, int(rrf_k))

    if method == FusionMethod.MAX:
        ordered = sorted(best_cosine.items(), key=lambda item: item[1], reverse=True)
    else:
        ordered = fuse_hits(hits, method, rrf_k=rrf_k)

    matching_ids = [tid for tid, _score in ordered]
    return MultiViewMergeResult(
        matching_ids=matching_ids,
        best_cosine=best_cosine,
        best_view=best_view,
        method=method,
        rrf_k=int(rrf_k),
    )


def fuse_hits(
    hits: Sequence[ScoredHit],
    method: FusionMethod | str,
    *,
    rrf_k: int = DEFAULT_RRF_K,
    view_weights: dict[str, float] | None = None,
) -> list[tuple[int, float]]:
    """
    Collapse per-vector hits to one score per tile_id.

    Returns (tile_id, fused_score) sorted descending.

    For RRF, each ``ScoredHit`` must carry a meaningful ``rank_in_list`` from
    its own view's FAISS list (do **not** MAX-merge first then re-rank — that
    makes RRF order-identical to MAX and only changes score scale).
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
        # Soft-max / log-sum-exp per tile (numerically stable). Does not reward
        # tiles merely for having more vectors the way raw sum(exp) would.
        by_tile: dict[int, list[float]] = defaultdict(list)
        for h in hits:
            by_tile[h.tile_id].append(float(h.score) * max(0.05, float(h.view_weight)))
        fused: dict[int, float] = {}
        for tid, vals in by_tile.items():
            arr = np.asarray(vals, dtype=np.float64)
            m = float(arr.max())
            fused[tid] = float(m + np.log(np.exp(arr - m).sum()))
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
