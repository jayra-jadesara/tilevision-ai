"""
Image processing utilities for TileVision AI.

Provides functions to validate image files, extract image metadata (size, dimensions),
and generate cached thumbnails.
"""

import hashlib
import logging
from pathlib import Path
from typing import Tuple
from PIL import Image

# Initialize module logger
logger = logging.getLogger("tilevision.image_utils")


def validate_image(image_path: Path) -> bool:
    """
    Verify if a file exists, is an image, and can be successfully loaded.

    Args:
        image_path: Absolute path to the image file.

    Returns:
        True if the file is a valid image, False otherwise.
    """
    if not image_path.exists() or not image_path.is_file():
        logger.warning(f"File does not exist or is not a file: {image_path}")
        return False

    try:
        with Image.open(image_path) as img:
            img.verify()  # Fast check to verify image integrity without loading data
        return True
    except (IOError, SyntaxError) as e:
        logger.warning(f"Failed to validate image structure for {image_path}: {e}")
        return False


def get_image_metadata(image_path: Path) -> Tuple[int, str]:
    """
    Retrieve image file size and dimensions.

    Args:
        image_path: Absolute path to the image file.

    Returns:
        A tuple of (file_size_in_bytes, dimensions_string_as_WxH).
        If dimensions cannot be fetched, returns (file_size_in_bytes, "UNKNOWN").
    """
    try:
        file_size = image_path.stat().st_size
    except OSError as e:
        logger.error(f"Failed to read file statistics for {image_path}: {e}")
        file_size = 0

    dimensions = "UNKNOWN"
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            dimensions = f"{width}x{height}"
    except Exception as e:
        logger.error(f"Failed to load image to extract dimensions for {image_path}: {e}")

    return file_size, dimensions


def generate_thumbnail(
    image_path: Path, thumbnail_dir: Path, size: Tuple[int, int] = (200, 200)
) -> Path:
    """
    Generate and save a cropped/resized thumbnail for the target image.
    
    Uses SHA-256 of the absolute image path to generate a unique filename
    to avoid file collisions and enable fast lookup cache.

    Args:
        image_path: Absolute path to the source image file.
        thumbnail_dir: Folder to store generated thumbnails.
        size: Desired thumbnail resolution (width, height).

    Returns:
        The absolute path to the generated thumbnail file.
    """
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    # Compute a unique hash of the path to avoid collision in cache directory
    path_hash = hashlib.sha256(str(image_path.resolve()).encode("utf-8")).hexdigest()
    thumb_path = thumbnail_dir / f"{path_hash}.jpg"

    # Cache hit check: Return existing thumbnail if it is valid
    if thumb_path.exists():
        return thumb_path

    try:
        with Image.open(image_path) as img:
            # Convert to RGB mode (in case of RGBA / CMYK) before saving as JPEG
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            
            # Use ImageOps.fit or thumbnail for maintaining aspect ratio.
            # Using thumbnail scales the image down so that it fits inside the specified bounding box.
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img.save(thumb_path, "JPEG", quality=85)
            logger.debug(f"Generated thumbnail for {image_path} -> {thumb_path}")
    except Exception as e:
        logger.error(f"Failed to generate thumbnail for {image_path}: {e}")
        # Return source path as fallback if thumbnail creation fails
        return image_path

    return thumb_path
