"""
Visual match score calibration for TileVision AI.

The hybrid reranker produces an internal ranking score in [0, 1].
This module maps that score to a user-facing "visual match %" that is
monotonic, bounded, and reserved for exact catalog matches.

CLIP text↔image cosine lives in a different numeric range — never reuse
the DINOv2 hybrid curve for description search.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def calibrate_display_percent(
    raw_score: float,
    *,
    exact_match: bool = False,
) -> float:
    """
    Map an internal hybrid ranking score to a display percentage.

    Parameters
    ----------
    raw_score
        Hybrid reranker final score in [0, 1].
    exact_match
        True when the candidate is a byte-identical catalog image.

    Returns
    -------
    float
        Visual match percentage in [0, 100].
    """
    if exact_match:
        return 100.0

    raw = max(0.0, min(1.0, raw_score))

    # Power curve: compress weak matches, preserve strong ones.
    # 0.50 raw -> ~41%, 0.70 -> ~58%, 0.85 -> ~72%, 0.95 -> ~84%
    mapped = 100.0 * (raw ** 1.35)

    # Reserve 100% exclusively for exact matches.
    return max(0.0, min(99.5, mapped))


# Piecewise anchors for CLIP text–image cosine (L2-normalized IP).
# Fitted 2026-10-14 against measured ClipEmbedder scores on a 12-image
# themed catalog + the production observation that "gray tile" top hits
# displayed ~17.8% under the DINOv2 curve ⇒ raw cosine ≈ 0.27
# (solve 100 * x**1.35 = 17.8). Measured cluster on synthetic tiles:
#   good query top-1 ≈ 0.21–0.24, unrelated "sushi" top-1 ≈ 0.16.
# Anchors lift ~0.27 → ~80% display while keeping ~0.16 in the low 20s.
_CLIP_DISPLAY_ANCHORS: Sequence[tuple[float, float]] = (
    (0.10, 0.0),
    (0.14, 12.0),
    (0.18, 32.0),
    (0.22, 55.0),
    (0.25, 72.0),
    (0.28, 84.0),
    (0.32, 92.0),
    (0.38, 96.0),
    # Keep headroom above the typical CLIP band so rare high cosines
    # (and test fakes) stay strictly monotonic instead of collapsing to 99.
    (0.50, 98.0),
    (0.70, 99.0),
    (1.00, 99.5),
)


def calibrate_clip_display_percent(raw_cosine: float) -> float:
    """
    Map a CLIP text↔image cosine similarity to a display percentage.

    Do **not** use :func:`calibrate_display_percent` for CLIP — that curve
    is tuned for the DINOv2 hybrid reranker, where 0.5 is a mid score.
    CLIP image–text cosines for good tile matches typically sit near
    0.25–0.35; the DINOv2 curve maps 0.27 → ~18% and triggers false
    "low confidence" banners.
    """
    raw = max(0.0, min(1.0, float(raw_cosine)))
    anchors = _CLIP_DISPLAY_ANCHORS
    if raw <= anchors[0][0]:
        return max(0.0, anchors[0][1])
    if raw >= anchors[-1][0]:
        return min(99.5, anchors[-1][1])
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x0 <= raw <= x1:
            t = (raw - x0) / max(1e-9, (x1 - x0))
            return max(0.0, min(99.5, y0 + t * (y1 - y0)))
    return max(0.0, min(99.5, 100.0 * raw))


# Distribution-aware description-search display (z-score vs per-query catalogue
# band, logistic-mapped). Tuned 2026-10-16 against customer log fixtures:
#   white  min/median/max 0.2362 / 0.2846 / 0.3041
#   gray   0.2300 / 0.2681 / 0.3096
#   dot    0.2318 / 0.2540 / 0.3022
#   texture 0.2406 / 0.2766 / 0.3047
CUSTOMER_LOG_FIXTURE_STATS: Dict[str, tuple[float, float, float]] = {
    "white": (0.2362, 0.2846, 0.3041),
    "gray": (0.2300, 0.2681, 0.3096),
    "dot": (0.2318, 0.2540, 0.3022),
    "texture": (0.2406, 0.2766, 0.3047),
}
_DESC_DISPLAY_LOGISTIC_SCALE = 1.15
_DESC_DISPLAY_LOGISTIC_CENTER = 0.65
_DESC_DISPLAY_STD_FLOOR = 0.008
_DESC_DISPLAY_MAX_PCT = 99.5
_DESC_DISPLAY_WORST_CAP = 40.0
# Top hit z-score below this ⇒ flat / weak separation for this query.
_DESC_LOW_CONFIDENCE_TOP_Z = 1.58


def _description_z_scores(catalogue_raw_scores: Sequence[float]) -> np.ndarray:
    arr = np.asarray([float(s) for s in catalogue_raw_scores], dtype=np.float64)
    if arr.size == 0:
        return arr
    mean = float(arr.mean())
    if arr.size == 1:
        return np.asarray([1.0], dtype=np.float64)
    std = float(arr.std(ddof=1))
    std = max(std, _DESC_DISPLAY_STD_FLOOR)
    return (arr - mean) / std


def description_search_top_z(catalogue_raw_scores: Sequence[float]) -> float:
    """Z-score of the best raw cosine in the deduped catalogue band."""
    z = _description_z_scores(catalogue_raw_scores)
    if z.size == 0:
        return 0.0
    return float(np.max(z))


def description_search_is_low_confidence(
    catalogue_raw_scores: Sequence[float],
    *,
    used_cutoff_fallback: bool = False,
) -> bool:
    """
    Weak description match: cutoff had to keep weak hits, or the top hit is
    not clearly above the per-query score cloud (low top z).
    """
    if used_cutoff_fallback:
        return True
    return description_search_top_z(catalogue_raw_scores) < _DESC_LOW_CONFIDENCE_TOP_Z


def calibrate_description_search_display_percent(
    raw_cosine: float,
    catalogue_raw_scores: Sequence[float],
) -> float:
    """
    Map one CLIP cosine to display % using the *query's* deduped score band.

    Monotonic in ``raw_cosine`` for a fixed catalogue distribution. Worst hits
    stay at or below ~40%. Do not use for DINOv2 image search.
    """
    mapped = calibrate_description_search_display_map(catalogue_raw_scores)
    if not mapped:
        return 0.0
    arr = np.asarray([float(s) for s in catalogue_raw_scores], dtype=np.float64)
    target = float(raw_cosine)
    # Stable tie-break: same raw score → same display (use first match index).
    for score, pct in zip(arr, mapped):
        if float(score) == target:
            return float(pct)
    # Fallback: interpolate in sorted raw order (monotonic).
    order = np.argsort(arr)
    sorted_scores = arr[order]
    sorted_pct = np.asarray(mapped, dtype=np.float64)[order]
    if target <= float(sorted_scores[0]):
        return float(sorted_pct[0])
    if target >= float(sorted_scores[-1]):
        return float(sorted_pct[-1])
    for i in range(len(sorted_scores) - 1):
        x0, x1 = float(sorted_scores[i]), float(sorted_scores[i + 1])
        if x0 <= target <= x1:
            y0, y1 = float(sorted_pct[i]), float(sorted_pct[i + 1])
            t = (target - x0) / max(1e-12, x1 - x0)
            return max(0.0, min(_DESC_DISPLAY_MAX_PCT, y0 + t * (y1 - y0)))
    return float(sorted_pct[-1])


def calibrate_description_search_display_map(
    catalogue_raw_scores: Sequence[float],
) -> List[float]:
    """
    Display % for each raw cosine in ``catalogue_raw_scores`` (same order).

    Preserves rank order; compresses the bottom of the band to <= 40%.
    """
    arr = np.asarray([float(s) for s in catalogue_raw_scores], dtype=np.float64)
    if arr.size == 0:
        return []
    if arr.size == 1:
        return [min(72.0, _DESC_DISPLAY_MAX_PCT)]

    z = _description_z_scores(arr)
    pct = 100.0 / (
        1.0
        + np.exp(
            -_DESC_DISPLAY_LOGISTIC_SCALE
            * (z - _DESC_DISPLAY_LOGISTIC_CENTER)
        )
    )
    pct = np.clip(pct, 0.0, _DESC_DISPLAY_MAX_PCT)

    order = np.argsort(arr)
    running = np.empty_like(pct)
    running[order[0]] = pct[order[0]]
    for j in range(1, len(order)):
        i = order[j]
        prev_i = order[j - 1]
        running[i] = max(float(pct[i]), float(running[prev_i]))
    pct = running

    hi = float(np.max(pct))
    lo = float(np.min(pct))
    if lo > _DESC_DISPLAY_WORST_CAP and hi > lo:
        # Affine squash so catalogue minimum maps to the worst-cap.
        scale = (_DESC_DISPLAY_WORST_CAP - hi) / (lo - hi)
        pct = hi + (pct - hi) * scale
        pct = np.clip(pct, 0.0, _DESC_DISPLAY_MAX_PCT)
        # Re-enforce monotonicity after squash.
        running[order[0]] = pct[order[0]]
        for j in range(1, len(order)):
            i = order[j]
            prev_i = order[j - 1]
            running[i] = max(float(pct[i]), float(running[prev_i]))
        pct = running

    return [round(float(x), 2) for x in pct]


def description_display_fixture_table(
    fixture_stats: Dict[str, tuple[float, float, float]],
    *,
    n_tiles: int = 38,
) -> Dict[str, Dict[str, float]]:
    """
    Build before/after display summary for customer-log fixture bands.

    ``fixture_stats`` maps query label → (min, median, max) raw cosines.
    """
    def _synth(min_s: float, med: float, max_s: float) -> List[float]:
        xs = np.linspace(min_s, max_s, n_tiles)
        xs = xs - float(np.median(xs)) + med
        xs = np.clip(xs, min_s, max_s)
        xs.sort()
        xs[0] = min_s
        xs[-1] = max_s
        return [float(x) for x in xs]

    out: Dict[str, Dict[str, float]] = {}
    for label, (lo, med, hi) in fixture_stats.items():
        scores = _synth(lo, med, hi)
        old_top = calibrate_clip_display_percent(max(scores))
        old_min = calibrate_clip_display_percent(min(scores))
        new_map = calibrate_description_search_display_map(scores)
        out[label] = {
            "top_old": old_top,
            "top_new": max(new_map),
            "min_old": old_min,
            "min_new": min(new_map),
            "top_z": description_search_top_z(scores),
        }
    return out
