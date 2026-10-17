"""
Fuse CLIP text ranks with tile colour labels when the query names a colour.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

from src.ai.search_quality.fusion import DEFAULT_RRF_K

# British + American spellings and synonyms.
_COLOUR_ALIASES: Dict[str, str] = {
    "gray": "gray",
    "grey": "gray",
    "white": "white",
    "black": "black",
    "beige": "beige",
    "cream": "cream",
    "brown": "brown",
    "blue": "blue",
    "green": "green",
    "red": "red",
    "yellow": "yellow",
}


def extract_colour_tokens(query: str) -> List[str]:
    """Return normalized colour tokens present in ``query`` (may be empty)."""
    q = (query or "").lower()
    if not q:
        return []
    tokens = set(re.findall(r"[a-z]+", q))
    out: List[str] = []
    for tok in tokens:
        canon = _COLOUR_ALIASES.get(tok)
        if canon and canon not in out:
            out.append(canon)
    return out


def _tile_matches_colour(tile_color: str, wanted: Sequence[str]) -> bool:
    if not wanted:
        return False
    label = (tile_color or "").strip().lower()
    if not label or label == "unknown":
        return False
    norm = "gray" if label == "grey" else label
    return norm in wanted


def fuse_clip_with_colour_rrf(
    scored_pairs: Sequence[Tuple[int, float]],
    tile_colors: Dict[int, str],
    colour_tokens: Sequence[str],
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> List[Tuple[int, float]]:
    """
    RRF merge of CLIP score order and colour-label order.

    Returns reordered ``(tile_id, raw_clip_score)``; scores are unchanged.
    Non-colour queries: pass empty ``colour_tokens`` → input order preserved.
    """
    if not scored_pairs:
        return []
    if not colour_tokens:
        return list(scored_pairs)

    clip_ranked = sorted(scored_pairs, key=lambda p: p[1], reverse=True)
    colour_hits = [
        (tid, sc)
        for tid, sc in clip_ranked
        if _tile_matches_colour(tile_colors.get(int(tid), ""), colour_tokens)
    ]
    if not colour_hits:
        return list(scored_pairs)

    colour_ranked = sorted(colour_hits, key=lambda p: p[1], reverse=True)

    rrf_scores: Dict[int, float] = {}
    for rank, (tid, _) in enumerate(clip_ranked, start=1):
        rrf_scores[int(tid)] = rrf_scores.get(int(tid), 0.0) + 1.0 / (
            rrf_k + rank
        )
    for rank, (tid, _) in enumerate(colour_ranked, start=1):
        rrf_scores[int(tid)] = rrf_scores.get(int(tid), 0.0) + 1.0 / (
            rrf_k + rank
        )

    # Stable tie-break: higher CLIP cosine wins.
    clip_score = {int(tid): float(sc) for tid, sc in scored_pairs}
    ordered_ids = sorted(
        rrf_scores.keys(),
        key=lambda tid: (rrf_scores[tid], clip_score.get(tid, 0.0)),
        reverse=True,
    )
    return [(tid, clip_score[tid]) for tid in ordered_ids if tid in clip_score]
