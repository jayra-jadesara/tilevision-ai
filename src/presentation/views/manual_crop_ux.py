"""
Manual Crop UX helpers (no silent override of the user's rectangle).

Manual Crop is user-controlled. These helpers only:
  - suggest a sensible *starting* box (aligned with Auto Crop), and
  - classify whether a selection is unusually tight so the UI can warn.

The final region is always whatever the user confirms — never replaced.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

# Match Auto/Precise over-crop keep threshold: below this, warn (do not override).
TIGHT_SELECTION_KEEP = 0.50


@dataclass(frozen=True, slots=True)
class ManualCropSuggestion:
    """Default selection in source-image pixel coordinates (left, top, right, bottom)."""

    box: tuple[int, int, int, int]
    method: str
    keep_ratio: float


def selection_keep_ratio(
    image_size: tuple[int, int],
    box: tuple[int, int, int, int],
) -> float:
    """Fraction of the source image covered by ``box`` (left, top, right, bottom)."""
    width, height = image_size
    src_area = max(1, width * height)
    left, top, right, bottom = box
    cw = max(0, right - left)
    ch = max(0, bottom - top)
    return float(cw * ch) / float(src_area)


def is_tight_selection(
    image_size: tuple[int, int],
    box: tuple[int, int, int, int],
    *,
    threshold: float = TIGHT_SELECTION_KEEP,
) -> bool:
    """True when the selection covers less than ``threshold`` of the source area."""
    return selection_keep_ratio(image_size, box) < threshold


def suggest_manual_default_box(image: Image.Image) -> ManualCropSuggestion:
    """
    Starting box for Manual Crop — same region Auto Crop would search today.

    Uses ``resolve_auto_tile_crop`` so clean / over-crop-guarded close-ups
    open as full-frame (users adjust down), while room photos open on the
    isolated tile surface (users can expand). Never applied as a silent
    override after the user draws their own rectangle.
    """
    from src.ai.preprocess.fast_tile_crop import resolve_auto_tile_crop

    rgb = image.convert("RGB")
    width, height = rgb.size
    resolved = resolve_auto_tile_crop(rgb)
    left, top, right, bottom = resolved.box
    # Clamp / normalize
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))
    box = (left, top, right, bottom)
    return ManualCropSuggestion(
        box=box,
        method=resolved.method,
        keep_ratio=selection_keep_ratio((width, height), box),
    )


def format_selection_status(
    image_size: tuple[int, int],
    box: tuple[int, int, int, int] | None,
) -> tuple[str, bool]:
    """
    Human-readable selection status for the Manual Crop dialog.

    Returns ``(label, is_warning)``.
    """
    if box is None:
        return ("No selection — drag a rectangle, or use the suggested starting region.", False)
    width, height = image_size
    left, top, right, bottom = box
    sw, sh = max(0, right - left), max(0, bottom - top)
    keep = selection_keep_ratio(image_size, box)
    pct = 100.0 * keep
    base = f"Selection: {sw}×{sh} px ({pct:.0f}% of image)"
    if is_tight_selection(image_size, box):
        return (
            f"{base} — small region may weaken edge/pattern match; expand if you meant the full tile.",
            True,
        )
    return (base, False)
