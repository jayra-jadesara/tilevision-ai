"""Compare UI search results against the expected customer dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExpectationResult:
    query_id: str
    expected_product: str
    top_product: str = ""
    top_path: str = ""
    rank_of_expected: Optional[int] = None
    ok: bool = False
    detail: str = ""
    scores: List[float] = field(default_factory=list)


def product_code_from_path(path: str) -> str:
    stem = Path(path).stem
    # catalog files are like TILE_A01_marble_beige.jpg
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "TILE":
        return parts[1]
    return stem


def evaluate_search(
    *,
    query_id: str,
    expected_product: str,
    result_paths: List[str],
    result_scores: Optional[List[float]] = None,
    max_acceptable_rank: int = 3,
) -> ExpectationResult:
    out = ExpectationResult(query_id=query_id, expected_product=expected_product)
    if result_scores:
        out.scores = list(result_scores)
    if not result_paths:
        out.detail = "No result paths displayed"
        return out

    products = [product_code_from_path(p) for p in result_paths]
    out.top_product = products[0]
    out.top_path = result_paths[0]
    try:
        out.rank_of_expected = products.index(expected_product) + 1
    except ValueError:
        out.rank_of_expected = None
        out.detail = (
            f"Expected product {expected_product!r} not in top-{len(products)} "
            f"(got {products[:5]})"
        )
        return out

    out.ok = out.rank_of_expected <= max_acceptable_rank
    if out.ok:
        out.detail = f"Expected {expected_product} at rank {out.rank_of_expected}"
    else:
        out.detail = (
            f"Expected {expected_product} within top {max_acceptable_rank}, "
            f"found at rank {out.rank_of_expected}"
        )
    return out


def evaluate_from_manifest(
    manifest: Dict[str, Any],
    query_id: str,
    result_paths: List[str],
) -> ExpectationResult:
    queries = {q["id"]: q for q in manifest.get("queries", [])}
    q = queries.get(query_id)
    if not q:
        return ExpectationResult(
            query_id=query_id,
            expected_product="?",
            detail=f"Unknown query id {query_id}",
        )
    return evaluate_search(
        query_id=query_id,
        expected_product=q["expected_product"],
        result_paths=result_paths,
        max_acceptable_rank=int(q.get("max_rank", 3)),
    )
