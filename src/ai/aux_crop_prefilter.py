"""
Cheap pre-embedding filter for index aux crops.

After DINOv2, ``_maybe_append_aux`` drops candidates with
``cos_vs_primary ≥ 0.985`` (or pair ≥ 0.99). Embedding those crops first
wastes ~20 % of DINOv2 time on typical catalogues (customer log: 12 of 30
aux crops discarded after full 3-scale embed).

This module predicts near-duplicates from **pixel** cues so we can skip
the DINOv2 pass. **Default is OFF** until validated on a real catalogue
with false-skips = 0 (see ``docs/INDEXING_SPEED_INVESTIGATION.md``).

Enable for experiments: ``TILEVISION_AUX_PREFILTER=1``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

logger = logging.getLogger("tilevision.ai.aux_crop_prefilter")

# Conservative: only skip when the downscaled crop is almost identical to
# the primary letterbox. Must be ≥ embedding drop threshold intent.
_DEFAULT_THUMB_COS_MIN = 0.995
_DEFAULT_AREA_FRAC_MIN = 0.70
_THUMB_EDGE = 32


@dataclass(frozen=True, slots=True)
class AuxPrefilterDecision:
    skip: bool
    reason: str
    thumb_cos: float
    area_frac: float


def aux_prefilter_enabled() -> bool:
    """Opt-in only — must stay false until false-skips = 0 on real tiles."""
    raw = os.environ.get("TILEVISION_AUX_PREFILTER", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _thumb_rgb_vector(image: Image.Image, edge: int = _THUMB_EDGE) -> np.ndarray:
    rgb = image.convert("RGB").resize((edge, edge), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm < 1e-6:
        return np.zeros_like(arr)
    return arr / norm


def crop_area_fraction(
    crop_box: tuple[int, int, int, int] | None,
    primary_size: tuple[int, int] | None,
) -> float:
    """Fraction of the primary frame covered by ``crop_box`` (0–1)."""
    if crop_box is None or primary_size is None:
        return 0.0
    pw, ph = primary_size
    full = max(1, int(pw) * int(ph))
    left, top, right, bottom = crop_box
    area = max(0, int(right) - int(left)) * max(0, int(bottom) - int(top))
    return float(area) / float(full)


def decide_aux_prefilter(
    primary_pil: Image.Image,
    crop_pil: Image.Image,
    *,
    crop_box: tuple[int, int, int, int] | None = None,
    primary_size: tuple[int, int] | None = None,
    thumb_cos_min: float = _DEFAULT_THUMB_COS_MIN,
    area_frac_min: float = _DEFAULT_AREA_FRAC_MIN,
) -> AuxPrefilterDecision:
    """
    Return whether to skip DINOv2 for this aux crop.

    Skips only when **both** area coverage and thumbnail cosine are high —
    a deliberately conservative AND so false-skips stay near zero.
    """
    area_frac = crop_area_fraction(crop_box, primary_size or primary_pil.size)
    a = _thumb_rgb_vector(primary_pil)
    b = _thumb_rgb_vector(crop_pil)
    thumb_cos = float(np.dot(a, b)) if a.size and b.size else 0.0

    if area_frac >= area_frac_min and thumb_cos >= thumb_cos_min:
        return AuxPrefilterDecision(
            skip=True,
            reason="thumb_near_dup",
            thumb_cos=thumb_cos,
            area_frac=area_frac,
        )
    return AuxPrefilterDecision(
        skip=False,
        reason="keep",
        thumb_cos=thumb_cos,
        area_frac=area_frac,
    )
