"""
Image preprocessing utilities for TileVision AI.

Centralizes image loading and preprocessing so DINOv2 and handcrafted
descriptors operate on the same representation.

Pipeline
--------
load -> EXIF transpose -> alpha composite -> optional border trim
     -> aspect-ratio-preserving letterbox resize -> OpenCV arrays
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.ai.models import PreprocessedImage

logger = logging.getLogger("tilevision.ai.image_preprocessor")

# DINOv2 ViT patch size is 14; 518 = 37 * 14.
TARGET_SIZE = 518

# Default max edge when decoding huge catalogue masters (70–200 MB files).
DEFAULT_MAX_DECODE_EDGE = 2048

# Neutral pad color — avoids biasing color descriptors toward white/black.
PAD_COLOR: Tuple[int, int, int] = (128, 128, 128)


@dataclass(slots=True)
class PreprocessConfig:
    """Runtime preprocessing limits (wired from AppSettings at startup)."""

    max_decode_edge: int = DEFAULT_MAX_DECODE_EDGE


class ImagePreprocessor:
    """Shared preprocessing pipeline for all AI feature extractors."""

    DEFAULT_SIZE: Tuple[int, int] = (TARGET_SIZE, TARGET_SIZE)
    _config = PreprocessConfig()

    @classmethod
    def configure(cls, *, max_decode_edge: int | None = None) -> None:
        """Update module-level decode limits (called once from app startup)."""
        if max_decode_edge is not None:
            cls._config.max_decode_edge = max(512, int(max_decode_edge))

    @classmethod
    def max_decode_edge(cls) -> int:
        return cls._config.max_decode_edge

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        max_decode_edge: int | None = None,
    ) -> Image.Image:
        """
        Load an image, applying early downscale for huge catalogue masters.

        Decodes to at most ``max_decode_edge`` on the longest side before
        border/crop/AI work — critical for 70–200 MB tile photography.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        max_edge = max_decode_edge or cls._config.max_decode_edge

        try:
            with Image.open(path) as probe:
                probe = ImageOps.exif_transpose(probe)
                width, height = probe.size

            if max(width, height) <= max_edge:
                with Image.open(path) as img:
                    return ImageOps.exif_transpose(img).copy()

            with Image.open(path) as img:
                if img.format in ("JPEG", "MPO", "WEBP") and hasattr(img, "draft"):
                    img.draft("RGB", (max_edge, max_edge))
                image = ImageOps.exif_transpose(img)
                image.thumbnail(
                    (max_edge, max_edge),
                    Image.Resampling.BICUBIC,
                )
                logger.debug(
                    "Early downscale %s: %dx%d -> %dx%d (max_edge=%d)",
                    path.name,
                    width,
                    height,
                    image.size[0],
                    image.size[1],
                    max_edge,
                )
                return image.copy()
        except Exception as e:
            logger.exception("Failed to load image: %s", path)
            raise RuntimeError(str(e)) from e

    @staticmethod
    def to_rgb(image: Image.Image) -> Image.Image:
        """
        Convert to RGB, compositing alpha onto a neutral background.
        """
        if image.mode in ("RGBA", "LA"):
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, PAD_COLOR)
            background.paste(rgba, mask=rgba.split()[-1])
            return background

        if image.mode == "P" and "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, PAD_COLOR)
            background.paste(rgba, mask=rgba.split()[-1])
            return background

        if image.mode != "RGB":
            return image.convert("RGB")

        return image

    @staticmethod
    def _is_uniform_border_row(row: np.ndarray, tolerance: int = 18) -> bool:
        """True when a row is a near-uniform light border (catalog background)."""
        if row.size == 0:
            return False
        mean = row.mean(axis=0)
        spread = np.max(np.abs(row.astype(np.int16) - mean.astype(np.int16)), axis=1)
        return bool(np.mean(spread <= tolerance) > 0.92 and mean.mean() > 200)

    @classmethod
    def trim_uniform_borders(
        cls,
        image: Image.Image,
        max_trim_ratio: float = 0.12,
    ) -> Image.Image:
        """
        Conservatively crop uniform white/light catalogue borders.

        Does nothing when no clear border is detected.
        """
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        if height < 32 or width < 32:
            return image

        max_v_trim = int(height * max_trim_ratio)
        max_h_trim = int(width * max_trim_ratio)

        top = 0
        for i in range(max_v_trim):
            if not cls._is_uniform_border_row(rgb[i]):
                break
            top = i + 1

        bottom = height
        for i in range(max_v_trim):
            if not cls._is_uniform_border_row(rgb[height - 1 - i]):
                break
            bottom = height - 1 - i

        left = 0
        for i in range(max_h_trim):
            if not cls._is_uniform_border_row(rgb[:, i]):
                break
            left = i + 1

        right = width
        for i in range(max_h_trim):
            if not cls._is_uniform_border_row(rgb[:, width - 1 - i]):
                break
            right = width - 1 - i

        if right - left < width * 0.5 or bottom - top < height * 0.5:
            return image

        if top == 0 and bottom == height and left == 0 and right == width:
            return image

        logger.debug(
            "Trimmed uniform borders: top=%d bottom=%d left=%d right=%d",
            top,
            bottom,
            left,
            right,
        )
        return image.crop((left, top, right, bottom))

    @classmethod
    def crop_to_content_region(
        cls,
        image: Image.Image,
        min_margin_ratio: float = 0.08,
    ) -> Image.Image:
        """
        Crop to the dominant textured region when clear background margins exist.

        Uses edge/variance detection — conservative: returns the original image
        when no confident content bounding box is found.
        """
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        if height < 48 or width < 48:
            return image

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        edges = cv2.Canny(blur, 40, 120)

        blur_f = blur.astype(np.float32)
        sq_blur = cv2.GaussianBlur(blur_f * blur_f, (15, 15), 0)
        variance = np.maximum(sq_blur - blur_f * blur_f, 0.0)
        texture_mask = (variance > 25.0).astype(np.uint8) * 255

        mask = cv2.bitwise_or(edges, texture_mask)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return image

        x, y, box_w, box_h = cv2.boundingRect(coords)
        if box_w < width * 0.35 or box_h < height * 0.35:
            return image

        margin_x = min(x, width - (x + box_w))
        margin_y = min(y, height - (y + box_h))
        if margin_x < width * min_margin_ratio and margin_y < height * min_margin_ratio:
            return image

        logger.debug(
            "Content-region crop: x=%d y=%d w=%d h=%d",
            x,
            y,
            box_w,
            box_h,
        )
        return image.crop((x, y, x + box_w, y + box_h))

    @classmethod
    def normalize_lighting(cls, image: Image.Image) -> Image.Image:
        """
        Mild LAB L-channel stretch for shadow/exposure differences.

        Only applied when the luminance dynamic range is clearly compressed,
        to avoid altering well-exposed catalogue images.
        """
        rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        low, high = np.percentile(l_channel, (2, 98))
        if high - low >= 40:
            return image

        stretched = np.clip(
            (l_channel.astype(np.float32) - low) * (255.0 / max(high - low, 1.0)),
            0,
            255,
        ).astype(np.uint8)
        merged = cv2.merge([stretched, a_channel, b_channel])
        corrected_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        corrected_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(corrected_rgb)

    @classmethod
    def _looks_like_scene_photo(cls, image: Image.Image) -> bool:
        """Heuristic: non-square framing or strong border/center difference."""
        width, height = image.size
        if width < 48 or height < 48:
            return False

        aspect = width / max(height, 1)
        if aspect < 0.88 or aspect > 1.14:
            return True

        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        margin_x = max(1, int(width * 0.12))
        margin_y = max(1, int(height * 0.12))
        if margin_x * 2 >= width or margin_y * 2 >= height:
            return False

        center = rgb[margin_y : height - margin_y, margin_x : width - margin_x]
        if center.size == 0:
            return False

        border_strips = [
            rgb[:margin_y, :],
            rgb[height - margin_y :, :],
            rgb[:, :margin_x],
            rgb[:, width - margin_x :],
        ]
        border = np.concatenate(
            [strip.reshape(-1, 3) for strip in border_strips],
            axis=0,
        )
        center_mean = center.reshape(-1, 3).mean(axis=0)
        border_mean = border.mean(axis=0)
        return float(np.linalg.norm(center_mean - border_mean)) >= 22.0

    @classmethod
    def focus_center_region(
        cls,
        image: Image.Image,
        ratio: float = 0.72,
    ) -> Image.Image:
        """Crop to the central region — common tile location in room photos."""
        width, height = image.size
        crop_w = max(1, int(width * ratio))
        crop_h = max(1, int(height * ratio))
        left = (width - crop_w) // 2
        top = (height - crop_h) // 2
        return image.crop((left, top, left + crop_w, top + crop_h))

    @classmethod
    def resize_letterbox(
        cls,
        image: Image.Image,
        target: int = TARGET_SIZE,
    ) -> Image.Image:
        """
        Resize preserving aspect ratio, then center-pad to a square canvas.
        """
        width, height = image.size
        if width <= 0 or height <= 0:
            return image

        scale = min(target / width, target / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        resized = image.resize(
            (new_width, new_height),
            Image.Resampling.BICUBIC,
        )

        canvas = Image.new("RGB", (target, target), PAD_COLOR)
        offset_x = (target - new_width) // 2
        offset_y = (target - new_height) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas

    @classmethod
    def resize(
        cls,
        image: Image.Image,
        size: Tuple[int, int] | None = None,
    ) -> Image.Image:
        """Backward-compatible alias for letterbox resize."""
        target = size[0] if size else TARGET_SIZE
        return cls.resize_letterbox(image, target=target)

    @staticmethod
    def to_numpy(image: Image.Image) -> np.ndarray:
        return np.asarray(image, dtype=np.uint8)

    @staticmethod
    def to_bgr(image: Image.Image) -> np.ndarray:
        rgb = np.asarray(image, dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @classmethod
    def preprocess(
        cls,
        image_path: str | Path,
    ) -> PreprocessedImage:
        image = cls.load(image_path)
        original_width, original_height = image.size

        image = cls.to_rgb(image)
        image = cls.trim_uniform_borders(image)
        image = cls.crop_to_content_region(image)
        image = cls.normalize_lighting(image)
        image = cls.resize_letterbox(image)

        rgb = cls.to_numpy(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        return PreprocessedImage(
            pil=image,
            rgb=rgb,
            bgr=bgr,
            gray=gray,
            width=original_width,
            height=original_height,
        )

    @classmethod
    def preprocess_for_query(
        cls,
        image_path: str | Path,
    ) -> PreprocessedImage:
        """
        Search-only preprocessing with extra handling for scene/room photos.

        Does not change the indexing pipeline or feature_version — only
        applied at query time to improve room-photo searches.
        """
        image = cls.load(image_path)
        original_width, original_height = image.size

        image = cls.to_rgb(image)
        image = cls.trim_uniform_borders(image)
        image = cls.crop_to_content_region(image, min_margin_ratio=0.05)

        if cls._looks_like_scene_photo(image):
            logger.debug("Scene photo detected — applying center focus crop.")
            image = cls.focus_center_region(image)

        image = cls.normalize_lighting(image)
        image = cls.resize_letterbox(image)

        rgb = cls.to_numpy(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        return PreprocessedImage(
            pil=image,
            rgb=rgb,
            bgr=bgr,
            gray=gray,
            width=original_width,
            height=original_height,
        )
