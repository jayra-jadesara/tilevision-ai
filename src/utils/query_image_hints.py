"""
Heuristics for query-image quality and search-result confidence.

Helps users understand when to crop a room photo and when catalog matches
are too weak to trust.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from src.core.models import SearchResult

# Display % below which a non-self match is considered weak (image / DINOv2).
LOW_CONFIDENCE_THRESHOLD = 35.0
MODERATE_CONFIDENCE_THRESHOLD = 55.0

# Description / CLIP display % thresholds (after calibrate_clip_display_percent).
# Slightly higher floor so a mid CLIP match (~55–70%) is not "low".
DESC_LOW_CONFIDENCE_THRESHOLD = 45.0
DESC_MODERATE_CONFIDENCE_THRESHOLD = 65.0

# Treat near-100% rows as the query tile itself (self-match).
_EXACT_MATCH_THRESHOLD = 99.5

_ASPECT_MIN = 0.88
_ASPECT_MAX = 1.14
_BORDER_FRACTION = 0.12
_BORDER_CENTER_MEAN_DIFF = 22.0


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    NONE = "none"


def should_suggest_crop(image_path: str | Path) -> tuple[bool, str]:
    """
    Return True when the query image likely contains scene clutter
    (room photos, furniture, non-square framing).
    """
    path = Path(image_path)
    if not path.exists():
        return False, ""

    # Cropped temp images from Crop & Search should not re-prompt.
    if "tilevision_crops" in path.as_posix().lower():
        return False, ""

    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            width, height = img.size
            if width < 1 or height < 1:
                return False, ""

            aspect = width / height
            if aspect < _ASPECT_MIN or aspect > _ASPECT_MAX:
                return True, (
                    "This photo is not square and may include walls, furniture, "
                    "or floor edges."
                )

            arr = np.asarray(img, dtype=np.float32)
            if _border_differs_from_center(arr):
                return True, (
                    "The edges of this photo look different from the center — "
                    "it may be a room/installation shot rather than a clean tile sample."
                )
    except OSError:
        return False, ""

    return False, ""


def _border_differs_from_center(rgb: np.ndarray) -> bool:
    height, width = rgb.shape[:2]
    margin_x = max(1, int(width * _BORDER_FRACTION))
    margin_y = max(1, int(height * _BORDER_FRACTION))

    if margin_x * 2 >= width or margin_y * 2 >= height:
        return False

    center = rgb[margin_y : height - margin_y, margin_x : width - margin_x]
    if center.size == 0:
        return False

    strips = [
        rgb[:margin_y, :],
        rgb[height - margin_y :, :],
        rgb[:, :margin_x],
        rgb[:, width - margin_x :],
    ]
    border = np.concatenate([strip.reshape(-1, 3) for strip in strips], axis=0)
    center_mean = center.reshape(-1, 3).mean(axis=0)
    border_mean = border.mean(axis=0)
    return float(np.linalg.norm(center_mean - border_mean)) >= _BORDER_CENTER_MEAN_DIFF


def best_non_self_score(results: Sequence[SearchResult]) -> Optional[float]:
    """Highest similarity % among results that are not exact self-matches."""
    for result in results:
        if result.similarity_score < _EXACT_MATCH_THRESHOLD:
            return float(result.similarity_score)
    return None


def classify_confidence(
    results: Sequence[SearchResult],
    *,
    search_kind: str = "image",
) -> ConfidenceLevel:
    """
    Classify result strength.

    ``search_kind`` is ``\"image\"`` (DINOv2 visual search) or
    ``\"description\"`` (CLIP text search) — thresholds differ because the
    display % curves are calibrated separately.
    """
    if not results:
        return ConfidenceLevel.NONE

    best = best_non_self_score(results)
    if best is None:
        return ConfidenceLevel.HIGH

    if search_kind == "description":
        low, moderate = DESC_LOW_CONFIDENCE_THRESHOLD, DESC_MODERATE_CONFIDENCE_THRESHOLD
    else:
        low, moderate = LOW_CONFIDENCE_THRESHOLD, MODERATE_CONFIDENCE_THRESHOLD

    if best < low:
        return ConfidenceLevel.LOW
    if best < moderate:
        return ConfidenceLevel.MODERATE
    return ConfidenceLevel.HIGH


def confidence_message(
    results: Sequence[SearchResult],
    *,
    search_kind: str = "image",
) -> Optional[str]:
    """
    User-facing guidance based on result strength.

    Description-search copy must never suggest Crop & Search (image-only).
    """
    if not results:
        if search_kind == "description":
            return (
                "No matching tiles for that description. Try a different phrase, "
                "or confirm the catalogue was rebuilt so CLIP embeddings exist."
            )
        return (
            "No similar tiles found. Add product photos of this tile to your "
            "indexed folder and scan again."
        )

    level = classify_confidence(results, search_kind=search_kind)
    best = best_non_self_score(results)

    if level == ConfidenceLevel.HIGH and best is None:
        return None

    if level == ConfidenceLevel.LOW:
        if search_kind == "description":
            return (
                "Low confidence — no strong description match in your catalog "
                f"(best alternative: {best:.0f}%). "
                "Try a different phrase (color, finish, style), or browse via "
                "Export Catalogue."
            )
        return (
            "Low confidence — no strong visual match in your catalog "
            f"(best alternative: {best:.0f}%). "
            "Try Crop & Search on the tile surface only, or add this tile as a "
            "clean product photo and re-index."
        )

    if level == ConfidenceLevel.MODERATE:
        if search_kind == "description":
            return (
                f"Moderate description match (best: {best:.0f}%). "
                "Scores are relative match ratings for text search — "
                "verify results visually."
            )
        return (
            f"Moderate match (best alternative: {best:.0f}%). "
            "Scores are visual match ratings, not literal accuracy — "
            "verify results visually. Only 100% means the exact same catalog image."
        )

    return None
