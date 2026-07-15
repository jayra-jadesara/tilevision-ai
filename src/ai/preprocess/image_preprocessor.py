"""
Image preprocessing utilities for TileVision AI.

This module centralizes all image loading and preprocessing logic so every
AI component (DINOv2, color descriptor, texture descriptor, edge descriptor)
operates on the exact same image representation.

Responsibilities
----------------
- Validate image path
- Load image safely
- Auto-rotate using EXIF
- Remove alpha channel
- Convert to RGB
- Resize for AI models
- Convert to OpenCV format
- Convert to NumPy
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

from src.ai.models import PreprocessedImage

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger("tilevision.ai.image_preprocessor")


class ImagePreprocessor:
    """
    Shared preprocessing pipeline for all AI feature extractors.
    """

    # DINOv2 works best on multiples of 14.
    DEFAULT_SIZE: Tuple[int, int] = (518, 518)

    @staticmethod
    def load(path: str | Path) -> Image.Image:
        """
        Load image from disk.

        Raises
        ------
        FileNotFoundError
            If file does not exist.

        RuntimeError
            If image cannot be opened.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        try:
            image = Image.open(path)

            # respect phone camera orientation
            image = ImageOps.exif_transpose(image)

            return image

        except Exception as e:
            logger.exception("Failed to load image: %s", path)
            raise RuntimeError(str(e)) from e

    @staticmethod
    def to_rgb(image: Image.Image) -> Image.Image:
        """
        Remove alpha channel and convert to RGB.
        """

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    @classmethod
    def resize(
        cls,
        image: Image.Image,
        size: Tuple[int, int] | None = None,
    ) -> Image.Image:
        """
        Resize image using bicubic interpolation.
        """

        if size is None:
            size = cls.DEFAULT_SIZE

        return image.resize(
            size,
            Image.Resampling.BICUBIC,
        )

    @staticmethod
    def to_numpy(image: Image.Image) -> np.ndarray:
        """
        Convert PIL image to RGB NumPy array.
        """

        return np.asarray(image, dtype=np.uint8)

    @staticmethod
    def to_bgr(image: Image.Image) -> np.ndarray:
        """
        Convert PIL image to OpenCV BGR image.
        """

        rgb = np.asarray(image, dtype=np.uint8)

        return cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        )

    @classmethod
    def preprocess(
        cls,
        image_path: str | Path,
    ) -> PreprocessedImage:

        image = cls.load(image_path)

        image = cls.to_rgb(image)

        image = cls.resize(image)

        rgb = cls.to_numpy(image)

        bgr = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        )

        gray = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2GRAY,
        )

        return PreprocessedImage(
            pil=image,
            rgb=rgb,
            bgr=bgr,
            gray=gray,
            width=image.width,
            height=image.height,
        )

