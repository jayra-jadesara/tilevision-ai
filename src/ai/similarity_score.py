"""
Visual match score calibration for TileVision AI.

The hybrid reranker produces an internal ranking score in [0, 1].
This module maps that score to a user-facing "visual match %" that is
monotonic, bounded, and reserved for exact catalog matches.

CLIP text↔image cosine lives in a different numeric range — never reuse
the DINOv2 hybrid curve for description search.
"""

from __future__ import annotations

from typing import Sequence


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
