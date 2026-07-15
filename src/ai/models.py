"""
AI data models for TileVision AI.

These models are shared across the AI engine.

Author:
TileVision AI v2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image


# ==========================================================
# Preprocessed Image
# ==========================================================

@dataclass(slots=True)
class PreprocessedImage:
    """
    Shared image representation.

    The image is loaded only once and reused by every AI module.

    Attributes
    ----------
    pil
        PIL image used by DINOv2.

    rgb
        RGB ndarray.

    bgr
        OpenCV BGR ndarray.

    gray
        OpenCV grayscale image.

    width
        Original image width.

    height
        Original image height.
    """

    pil: Image.Image

    rgb: np.ndarray

    bgr: np.ndarray

    gray: np.ndarray

    width: int

    height: int


# ==========================================================
# Tile Features
# ==========================================================

@dataclass(slots=True)
class TileFeatures:
    """
    Complete AI representation of a tile.

    Combines:
    - DINOv2 semantic/visual embedding
    - HSV color distribution
    - LBP texture information
    - Edge structure
    - Pattern descriptor for repeated/local tile patterns
    - Dominant color
    """

    embedding: np.ndarray

    color_histogram: np.ndarray

    texture_histogram: np.ndarray

    edge_histogram: np.ndarray

    pattern_features: np.ndarray

    dominant_color: Tuple[int, int, int]

    width: int

    height: int


# ==========================================================
# Search Score
# ==========================================================

@dataclass(slots=True)
class SearchScore:
    """
    Detailed score used for reranking.
    """

    embedding: float

    color: float

    texture: float

    edge: float

    pattern: float

    final: float