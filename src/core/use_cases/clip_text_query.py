"""
CLIP text query helpers: prompt templates + relevance cutoff.

Template ensembling improves single-word queries (e.g. ``gray``) which are
CLIP's weakest case. Cutoff avoids dumping top_k of an unfiltered catalogue.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# Keep raw text as one member so free-form phrases are not diluted.
_TEXT_TEMPLATES: tuple[str, ...] = (
    "{q}",
    "a photo of a {q} tile",
    "a close-up photo of {q} ceramic floor tile texture",
    "{q} tile",
)

# Absolute cosine floor (CLIP IP after L2). Soft gate for near-zero hits.
CLIP_ABS_FLOOR = 0.24
# Floor on the adaptive top-margin so flat catalogues still prune a little.
CLIP_MIN_MARGIN = 0.02
# Fraction of (top − median) used as the adaptive margin.
# Why not a fixed 0.10: real CLIP image–text cosines on a small catalogue
# cluster in a narrow band (~0.18–0.30). A fixed 0.10 margin is often larger
# than the whole catalogue spread, so matched_pre_cutoff ≈ catalogue size.
# Scaling to 0.35 × (top − median) tracks that band; CLIP_MIN_MARGIN avoids
# over-pruning when top ≈ median (noisy ties / flat scores).
CLIP_SPREAD_FRACTION = 0.35
# Never return an empty screen — always keep at least this many.
CLIP_MIN_KEEP = 3


def description_prompt_variants(query: str) -> List[str]:
    """Return unique prompt strings for ensembling (raw + templates)."""
    q = (query or "").strip()
    if not q:
        return []
    seen: set[str] = set()
    out: List[str] = []
    for tmpl in _TEXT_TEMPLATES:
        text = tmpl.format(q=q).strip()
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def average_l2_normalized(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Mean of L2-normalized vectors, then re-normalize."""
    if not vectors:
        raise ValueError("No vectors to average.")
    stacked = np.stack(
        [np.asarray(v, dtype=np.float32).reshape(-1) for v in vectors],
        axis=0,
    )
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("Averaged CLIP text embedding is empty.")
    return (mean / norm).astype(np.float32)


def distribution_aware_margin(top: float, median: float) -> float:
    """
    Adaptive margin below the top score from the score distribution.

    ``max(CLIP_MIN_MARGIN, CLIP_SPREAD_FRACTION * (top - median))``.
    """
    spread = max(0.0, float(top) - float(median))
    return max(CLIP_MIN_MARGIN, CLIP_SPREAD_FRACTION * spread)


def score_distribution_stats(
    scores: Sequence[float],
    *,
    abs_floor: float = CLIP_ABS_FLOOR,
    top_margin: float | None = None,
) -> Dict[str, object]:
    """
    Summarize raw CLIP cosines for logging / tuning.

    Returns top-10 (desc), min, max, median, spread (max−min), counts above
    floor / within margin. Does not include file names.
    """
    arr = [float(s) for s in scores]
    if not arr:
        return {
            "count": 0,
            "top10": [],
            "min": None,
            "max": None,
            "median": None,
            "spread": None,
            "above_floor": 0,
            "within_margin": 0,
            "margin": None,
        }
    ordered = sorted(arr, reverse=True)
    top = ordered[0]
    lo = ordered[-1]
    med = float(np.median(np.asarray(ordered, dtype=np.float64)))
    margin = (
        float(top_margin)
        if top_margin is not None
        else distribution_aware_margin(top, med)
    )
    above_floor = sum(1 for s in ordered if s >= abs_floor)
    within_margin = sum(1 for s in ordered if s >= (top - margin))
    return {
        "count": len(ordered),
        "top10": [round(s, 4) for s in ordered[:10]],
        "min": round(lo, 4),
        "max": round(top, 4),
        "median": round(med, 4),
        "spread": round(top - lo, 4),
        "above_floor": above_floor,
        "within_margin": within_margin,
        "margin": round(margin, 4),
    }


def apply_relevance_cutoff(
    scored: Sequence[Tuple[int, float]],
    *,
    abs_floor: float = CLIP_ABS_FLOOR,
    min_keep: int = CLIP_MIN_KEEP,
    top_k: int = 20,
) -> Tuple[List[Tuple[int, float]], bool, int]:
    """
    Filter (id, raw_score) pairs by absolute floor and distribution-aware
    margin from the top.

    Returns ``(kept, used_fallback_min, catalogue_hits_before_cutoff)``.
    ``used_fallback_min`` is True when we had to keep weak results to avoid
    an empty screen (caller should show low-confidence banner).
    """
    if not scored:
        return [], False, 0
    ordered = sorted(scored, key=lambda p: p[1], reverse=True)
    before = len(ordered)
    scores = [float(sc) for _, sc in ordered]
    top = scores[0]
    median = float(np.median(np.asarray(scores, dtype=np.float64)))
    top_margin = distribution_aware_margin(top, median)
    kept = [
        (tid, sc)
        for tid, sc in ordered
        if sc >= abs_floor and sc >= (top - top_margin)
    ]
    used_fallback = False
    if len(kept) < min_keep:
        kept = list(ordered[: min(min_keep, len(ordered))])
        used_fallback = True
    if len(kept) > top_k:
        kept = kept[:top_k]
    return kept, used_fallback, before
