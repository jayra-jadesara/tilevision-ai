"""
Visual match score calibration for TileVision AI.

The hybrid reranker produces an internal ranking score in [0, 1].
This module maps that score to a user-facing "visual match %" that is
monotonic, bounded, and reserved for exact catalog matches.
"""

from __future__ import annotations


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
