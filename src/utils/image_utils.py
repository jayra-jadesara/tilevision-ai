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

# Feature 1 (Folder Indexing) supported formats, per product requirements.
# Shared by both the folder-scan indexer and the real-time folder watcher
# (monitor_folder.py) so the two can never drift out of sync on which
# formats are supported.
from src.utils.image_formats import register_optional_image_formats, supported_image_extensions

register_optional_image_formats()
SUPPORTED_IMAGE_EXTENSIONS = supported_image_extensions()


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


def get_thumbnail_path(image_path: Path, thumbnail_dir: Path) -> Path:
    """
    Compute the deterministic cached-thumbnail file path for a given source
    image, WITHOUT touching the filesystem (pure path calculation).

    Uses SHA-256 of the resolved absolute image path as the filename, so the
    same source image always maps to the same thumbnail path. Shared by
    generate_thumbnail() (which creates the file) and the search use case
    (which looks it up) so the hashing logic can never drift out of sync
    between the two call sites.

    Args:
        image_path: Absolute (or resolvable) path to the source image file.
        thumbnail_dir: Folder where thumbnails are cached.

    Returns:
        The deterministic absolute path where this image's thumbnail is
        (or would be) stored. Does not guarantee the file exists.
    """
    path_hash = hashlib.sha256(str(Path(image_path).resolve()).encode("utf-8")).hexdigest()
    return Path(thumbnail_dir) / f"{path_hash}.jpg"


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

    thumb_path = get_thumbnail_path(image_path, thumbnail_dir)

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


def compute_sha256(file_path: Path) -> str:
    """
    Compute the SHA-256 cryptographic hash of a file.

    Args:
        file_path: Path to the target file.

    Returns:
        Hexadecimal SHA-256 hash string.
    """
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    except OSError as e:
        logger.error(f"Failed to read file for SHA-256 hashing at {file_path}: {e}")
        return ""


def compute_dhash(image_path: Path) -> str:
    """
    Compute a 64-bit difference hash (dHash) for an image.
    
    Resizes image to 9x8, converts to grayscale, and compares adjacent pixels.
    Returns a 16-character hexadecimal string representing the hash.

    Args:
        image_path: Path to the target image file.

    Returns:
        A 16-character hexadecimal string dHash.
    """
    try:
        with Image.open(image_path) as img:
            # Resize to 9x8, converting to grayscale
            img_gray = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
            pixels = list(img_gray.getdata())
            
            # Difference calculation
            diff = []
            for row in range(8):
                for col in range(8):
                    pixel_left = pixels[row * 9 + col]
                    pixel_right = pixels[row * 9 + col + 1]
                    diff.append(pixel_left > pixel_right)
                    
            # Convert binary list to hex string
            decimal_value = 0
            for index, value in enumerate(diff):
                if value:
                    decimal_value += 1 << index
                    
            # Format as 16-character hex padded with leading zeros
            return f"{decimal_value:016x}"
    except Exception as e:
        logger.error(f"Failed to compute dHash for {image_path}: {e}")
        return ""


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """
    Compute the Hamming distance between two hex-encoded perceptual hashes
    (e.g. two compute_dhash() outputs). Lower values mean more visually
    similar images (0 = identical hash). Used by Feature 5 (Duplicate
    Detection) to cluster near-duplicate tiles.

    Args:
        hash_a: First 16-char hex hash string.
        hash_b: Second 16-char hex hash string.

    Returns:
        The number of differing bits, or -1 if either hash is invalid/empty.
    """
    if not hash_a or not hash_b:
        return -1
    try:
        return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")
    except ValueError:
        return -1
