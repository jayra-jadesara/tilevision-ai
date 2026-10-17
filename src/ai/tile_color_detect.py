"""
Deterministic tile colour labels from image pixels (HSV heuristics).

Used for metadata backfill and description-search colour-word fusion eval.
Filename metadata always wins at index time when it parses a colour.
"""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

PathLike = Union[str, Path]

# Canonical labels stored on TileImage.color (title case).
DETECTABLE_COLORS = frozenset(
    {
        "White",
        "Black",
        "Gray",
        "Beige",
        "Cream",
        "Brown",
        "Blue",
        "Green",
        "Red",
        "Yellow",
    }
)


def detect_color_from_image(path: PathLike) -> str:
    """
    Label dominant colour from resized RGB pixels.

    Returns ``Unknown`` when no rule matches (caller may leave DB unchanged).
    """
    p = Path(path)
    try:
        img = Image.open(p).convert("RGB").resize((64, 64))
    except Exception:
        return "Unknown"
    arr = np.asarray(img, dtype=np.float32) / 255.0
    hs, ss, vs = [], [], []
    for r, g, b in arr.reshape(-1, 3):
        h, s, v = colorsys.rgb_to_hsv(float(r), float(g), float(b))
        hs.append(h)
        ss.append(s)
        vs.append(v)
    s_med = float(np.median(ss))
    v_med = float(np.median(vs))
    h_med = float(np.median(hs))

    if s_med < 0.18:
        if v_med >= 0.78:
            return "White"
        if v_med <= 0.22:
            return "Black"
        if v_med >= 0.68:
            return "Cream"
        return "Gray"

    # Chromatic hues (deterministic bucket order).
    if s_med >= 0.22:
        if 0.48 <= h_med <= 0.68:
            return "Blue"
        if 0.22 <= h_med < 0.48:
            return "Green"
        if h_med < 0.08 or h_med > 0.92:
            return "Red"
        if 0.12 <= h_med <= 0.18:
            return "Yellow"

    warm = h_med < 0.15 or h_med > 0.90 or (0.08 <= h_med <= 0.18)
    if warm and s_med < 0.45:
        if v_med >= 0.62:
            return "Beige"
        if v_med >= 0.50:
            return "Cream"
        return "Brown"
    if s_med < 0.35 and 0.35 <= v_med <= 0.70:
        return "Gray"
    return "Unknown"


def resolve_index_color(
    filename_color: str,
    image_path: PathLike,
) -> tuple[str, str]:
    """
    Return ``(color, color_source)`` for indexing.

    ``color_source`` is ``filename`` when parse succeeded, else ``auto`` when
    pixels produced a label, else ``unknown``.
    """
    parsed = (filename_color or "").strip()
    if parsed and parsed.lower() != "unknown":
        return parsed, "filename"
    detected = detect_color_from_image(image_path)
    if detected != "Unknown":
        return detected, "auto"
    return "Unknown", "unknown"
