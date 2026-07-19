"""
Shared image format registration for TileVision AI.

Registers optional formats (HEIC/HEIF) so Mac showroom photos from iPhones
work the same as JPEG/PNG on Windows.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("tilevision.image_formats")

_BASE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_HEIC_EXTENSIONS = frozenset({".heic", ".heif"})
_heif_registered = False


def register_optional_image_formats() -> None:
    """Register HEIC/HEIF with Pillow when pillow-heif is installed."""
    global _heif_registered
    if _heif_registered:
        return
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        _heif_registered = True
        logger.debug("HEIC/HEIF support enabled via pillow-heif.")
    except ImportError:
        logger.debug("pillow-heif not installed — HEIC/HEIF files are skipped.")


def supported_image_extensions() -> frozenset[str]:
    """Extensions accepted for folder indexing and folder watch."""
    register_optional_image_formats()
    if _heif_registered:
        return _BASE_EXTENSIONS | _HEIC_EXTENSIONS
    return _BASE_EXTENSIONS


def query_image_extensions() -> frozenset[str]:
    """Extensions accepted for visual search query images."""
    register_optional_image_formats()
    extra = {".bmp", ".gif", ".tiff", ".tif"}
    if _heif_registered:
        extra |= _HEIC_EXTENSIONS
    return _BASE_EXTENSIONS | extra
