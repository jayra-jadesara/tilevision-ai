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

# Display % below which a non-self match is considered weak.
LOW_CONFIDENCE_THRESHOLD = 35.0
MODERATE_CONFIDENCE_THRESHOLD = 55.0

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


def classify_confidence(results: Sequence[SearchResult]) -> ConfidenceLevel:
    if not results:
        return ConfidenceLevel.NONE

    best = best_non_self_score(results)
    if best is None:
        return ConfidenceLevel.HIGH

    if best < LOW_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.LOW
    if best < MODERATE_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.MODERATE
    return ConfidenceLevel.HIGH


def confidence_message(results: Sequence[SearchResult]) -> Optional[str]:
    """
    User-facing guidance based on result strength.
    """
    if not results:
        return (
            "No similar tiles found. Add product photos of this tile to your "
            "indexed folder and scan again."
        )

    level = classify_confidence(results)
    best = best_non_self_score(results)

    if level == ConfidenceLevel.HIGH and best is None:
        return None

    if level == ConfidenceLevel.LOW:
        return (
            "Low confidence — no strong visual match in your catalog "
            f"(best alternative: {best:.0f}%). "
            "Try Crop & Search on the tile only, or add this tile as a "
            "clean product photo and re-index."
        )

    if level == ConfidenceLevel.MODERATE:
        return (
            f"Moderate match (best alternative: {best:.0f}%). "
            "Please verify results visually. Crop & Search may help for room photos."
        )

    return None
