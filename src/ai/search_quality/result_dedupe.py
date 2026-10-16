"""
Collapse catalogue copies that share the same image content into one result row.

Used by image, multi-photo, and description search *before* top_k and before
cutoff statistics so triplicated files do not inflate counts or score bands.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List, Optional, Sequence, Tuple

from src.core.models import SearchResult, TileImage
from src.utils.image_utils import hamming_distance

# Align with search_tiles near-exact dHash threshold (same content family).
NEAR_EXACT_DHASH_THRESHOLD = 3


def content_key(tile: TileImage) -> str:
    """
    Stable key for grouping identical / near-identical catalogue files.

    Prefer sha256; fall back to perceptual hash; else unique per-id so we
    never wrongly merge distinct tiles.
    """
    sha = (getattr(tile, "sha256_hash", None) or "").strip().lower()
    if sha:
        return f"sha:{sha}"
    ph = (getattr(tile, "perceptual_hash", None) or "").strip().lower()
    if ph:
        return f"ph:{ph}"
    tid = getattr(tile, "id", None)
    if tid is not None:
        return f"id:{int(tid)}"
    return f"path:{(tile.file_path or '').strip().lower()}"


def _keys_are_near_exact(a: str, b: str) -> bool:
    if a == b:
        return True
    if a.startswith("ph:") and b.startswith("ph:"):
        try:
            return hamming_distance(a[3:], b[3:]) <= NEAR_EXACT_DHASH_THRESHOLD
        except Exception:
            return False
    return False


def collapse_scored_by_content(
    scored: Sequence[Tuple[TileImage, float]],
) -> List[Tuple[TileImage, float, List[str]]]:
    """
    Keep the highest-scoring tile per content key.

    Returns ``(primary_tile, score, all_paths)`` with ``all_paths[0]`` the
    primary path and the rest the other copies (same score group may include
    weaker same-hash hits that lost to the primary).
    """
    # First pass: best score per exact key.
    best: dict[str, Tuple[TileImage, float]] = {}
    members: dict[str, List[TileImage]] = {}
    order: List[str] = []

    for tile, score in scored:
        key = content_key(tile)
        # Merge into an existing near-exact perceptual group when sha missing.
        merge_into: Optional[str] = None
        if key.startswith("ph:"):
            for existing in order:
                if _keys_are_near_exact(key, existing):
                    merge_into = existing
                    break
        use_key = merge_into or key
        if use_key not in best:
            best[use_key] = (tile, float(score))
            members[use_key] = [tile]
            order.append(use_key)
            continue
        members[use_key].append(tile)
        prev_tile, prev_score = best[use_key]
        if float(score) > prev_score:
            best[use_key] = (tile, float(score))

    out: List[Tuple[TileImage, float, List[str]]] = []
    for key in order:
        primary, score = best[key]
        paths: List[str] = []
        seen: set[str] = set()
        # Primary first.
        p0 = str(primary.file_path or "")
        if p0:
            paths.append(p0)
            seen.add(p0)
        for t in members[key]:
            p = str(t.file_path or "")
            if p and p not in seen:
                paths.append(p)
                seen.add(p)
        out.append((primary, score, paths))
    # Preserve descending score order among unique tiles.
    out.sort(key=lambda row: row[1], reverse=True)
    return out


def collapse_search_results(
    results: Sequence[SearchResult],
    *,
    top_k: Optional[int] = None,
) -> List[SearchResult]:
    """
    Dedupe ``SearchResult`` rows by content; attach ``copy_paths``.

    Score order of unique tiles is preserved (highest score wins the row).
    """
    scored = [(r.tile, float(r.similarity_score)) for r in results]
    # Rebuild using raw similarity on tile — callers that need raw cosine
    # should pass pre-collapsed lists; here we keep display scores stable.
    # Prefer original SearchResult objects so ORB / thumbnails survive.
    by_key: dict[str, SearchResult] = {}
    members: dict[str, List[SearchResult]] = {}
    order: List[str] = []

    for result in results:
        key = content_key(result.tile)
        merge_into: Optional[str] = None
        if key.startswith("ph:"):
            for existing in order:
                if _keys_are_near_exact(key, existing):
                    merge_into = existing
                    break
        use_key = merge_into or key
        if use_key not in by_key:
            by_key[use_key] = result
            members[use_key] = [result]
            order.append(use_key)
            continue
        members[use_key].append(result)
        if float(result.similarity_score) > float(by_key[use_key].similarity_score):
            by_key[use_key] = result

    collapsed: List[SearchResult] = []
    for key in order:
        primary = by_key[key]
        paths: List[str] = []
        seen: set[str] = set()
        p0 = str(primary.tile.file_path or "")
        if p0:
            paths.append(p0)
            seen.add(p0)
        for r in members[key]:
            p = str(r.tile.file_path or "")
            if p and p not in seen:
                paths.append(p)
                seen.add(p)
        collapsed.append(
            replace(
                primary,
                copy_paths=paths,
                extra_copy_count=max(0, len(paths) - 1),
            )
        )

    collapsed.sort(key=lambda r: float(r.similarity_score), reverse=True)
    if top_k is not None and top_k > 0:
        collapsed = collapsed[: int(top_k)]
    return collapsed


def unique_and_file_counts(tiles: Iterable[TileImage]) -> Tuple[int, int]:
    """Return ``(unique_content_count, total_file_count)``."""
    keys: set[str] = set()
    total = 0
    for tile in tiles:
        total += 1
        keys.add(content_key(tile))
    return len(keys), total
