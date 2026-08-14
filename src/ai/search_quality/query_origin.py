"""
Query-origin hint for search preprocessing.

Drop-search (no hint) keeps the existing QueryAnalyzer routing.
Crop-tool searches (Auto / Precise / Manual) pass ``crop_tool`` so an
already-isolated tile is not re-guessed as ``clean_tile`` with a single
straightened view.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class QueryOrigin(str, Enum):
    AUTO = "auto"
    CROP_TOOL = "crop_tool"


_CROP_NAME_PREFIXES = ("autocrop_", "precise_")
_MANUAL_CROP_PREFIX = "crop_"


def resolve_query_origin(
    image_path: str | Path | None,
    explicit: str | QueryOrigin | None = None,
) -> QueryOrigin:
    """
    Resolve origin from an explicit hint, else from the query path.

    Default (no hint, ordinary path) is ``AUTO`` — unchanged drop-search.
    """
    if explicit is not None and str(explicit).strip():
        value = str(explicit).strip().lower()
        if value in {QueryOrigin.CROP_TOOL.value, "crop", "crop_tool"}:
            return QueryOrigin.CROP_TOOL
        if value in {QueryOrigin.AUTO.value, "drop"}:
            return QueryOrigin.AUTO

    if image_path is None:
        return QueryOrigin.AUTO

    path = Path(image_path)
    posix = path.as_posix().lower()
    if "tilevision_crops" in posix:
        return QueryOrigin.CROP_TOOL
    name = path.name.lower()
    if name.startswith(_CROP_NAME_PREFIXES):
        return QueryOrigin.CROP_TOOL
    if name.startswith(_MANUAL_CROP_PREFIX):
        stem = path.stem
        remainder = stem[len(_MANUAL_CROP_PREFIX) :]
        if "_" in remainder:
            _base, suffix = remainder.rsplit("_", 1)
            if suffix.isdigit():
                return QueryOrigin.CROP_TOOL
    return QueryOrigin.AUTO
