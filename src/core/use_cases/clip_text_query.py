"""
CLIP text query helpers: prompt templates + relevance cutoff.

Template ensembling improves single-word queries (e.g. ``gray``) which are
CLIP's weakest case. Cutoff avoids dumping top_k of an unfiltered catalogue.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

# Keep raw text as one member so free-form phrases are not diluted.
_TEXT_TEMPLATES: tuple[str, ...] = (
    "{q}",
    "a photo of a {q} tile",
    "a close-up photo of {q} ceramic floor tile texture",
    "{q} tile",
)

# Absolute cosine floor (CLIP IP after L2). Tuned lightly on colour-word eval;
# reported as such in the PR — not overfit on the same words without disclosure.
CLIP_ABS_FLOOR = 0.24
# Keep results within this margin of the top raw score.
CLIP_TOP_MARGIN = 0.10
# Never return an empty screen — always keep at least this many.
CLIP_MIN_KEEP = 3


def description_prompt_variants(query: str) -> List[str]:
    """Return unique prompt strings for ensembling (raw + templates)."""
    q = (query or "").strip()
    if not q:
        return []
    seen: set[str] = set()
    out: List[str] = []
    for tmpl in _TEXT_TEMPLATES:
        text = tmpl.format(q=q).strip()
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def average_l2_normalized(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Mean of L2-normalized vectors, then re-normalize."""
    if not vectors:
        raise ValueError("No vectors to average.")
    stacked = np.stack(
        [np.asarray(v, dtype=np.float32).reshape(-1) for v in vectors],
        axis=0,
    )
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("Averaged CLIP text embedding is empty.")
    return (mean / norm).astype(np.float32)


def apply_relevance_cutoff(
    scored: Sequence[Tuple[int, float]],
    *,
    abs_floor: float = CLIP_ABS_FLOOR,
    top_margin: float = CLIP_TOP_MARGIN,
    min_keep: int = CLIP_MIN_KEEP,
    top_k: int = 20,
) -> Tuple[List[Tuple[int, float]], bool, int]:
    """
    Filter (id, raw_score) pairs by absolute floor and margin from the top.

    Returns ``(kept, used_fallback_min, catalogue_hits_before_cutoff)``.
    ``used_fallback_min`` is True when we had to keep weak results to avoid
    an empty screen (caller should show low-confidence banner).
    """
    if not scored:
        return [], False, 0
    ordered = sorted(scored, key=lambda p: p[1], reverse=True)
    before = len(ordered)
    top = float(ordered[0][1])
    kept = [
        (tid, sc)
        for tid, sc in ordered
        if sc >= abs_floor and sc >= (top - top_margin)
    ]
    used_fallback = False
    if len(kept) < min_keep:
        kept = list(ordered[: min(min_keep, len(ordered))])
        used_fallback = True
    if len(kept) > top_k:
        kept = kept[:top_k]
    return kept, used_fallback, before
