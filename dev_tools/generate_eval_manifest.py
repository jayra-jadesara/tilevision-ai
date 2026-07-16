"""
Generate eval/my_queries.jsonl from the indexed catalog.

Uses embedding similarity to group visually related tiles and writes a
ground-truth manifest for eval_recall_precision.py.

Usage:
    python dev_tools/generate_eval_manifest.py
    python dev_tools/generate_eval_manifest.py --output eval/my_queries.jsonl --min-similarity 0.72
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.pattern_classifier import PatternClassifier
from src.config.settings import AppSettings
from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteImageRepository

# Visual groups manually curated for the default showroom catalog.
# Used as fallback when embedding similarity is ambiguous.
_CURATED_GROUPS: dict[str, list[int]] = {
    "Speckled": [2, 8, 18, 38],
    "Geometric": [4, 5],
    "Marble": [3, 36],
    "Plain": [6],
}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def _build_groups_by_embedding(
    repo: SQLiteImageRepository,
    min_similarity: float,
) -> dict[int, set[int]]:
    tiles = [
        tile
        for tile in repo.get_all()
        if tile.is_indexed and tile.id is not None and tile.features is not None
    ]
    groups: dict[int, set[int]] = {}

    for query in tiles:
        assert query.id is not None
        relevant: set[int] = set()
        q_emb = query.features.embedding
        for candidate in tiles:
            if candidate.id == query.id:
                continue
            sim = _cosine(q_emb, candidate.features.embedding)
            if sim >= min_similarity:
                relevant.add(candidate.id)
        groups[query.id] = relevant

    return groups


def _merge_curated_groups(repo: SQLiteImageRepository) -> dict[int, set[int]]:
    id_by_group: dict[int, set[int]] = {}
    tiles = {tile.id: tile for tile in repo.get_all() if tile.id is not None}

    for category, ids in _CURATED_GROUPS.items():
        present = [tile_id for tile_id in ids if tile_id in tiles]
        for tile_id in present:
            others = {other for other in present if other != tile_id}
            id_by_group[tile_id] = id_by_group.get(tile_id, set()) | others

    return id_by_group


def generate_manifest(
    output_path: Path,
    *,
    min_similarity: float,
    use_curated: bool,
) -> int:
    settings = AppSettings()
    repo = SQLiteImageRepository(db_context=DatabaseContext(db_path=settings.database_path))

    if use_curated:
        groups = _merge_curated_groups(repo)
    else:
        groups = _build_groups_by_embedding(repo, min_similarity)

    lines: list[str] = []
    for tile in repo.get_all():
        if not tile.is_indexed or tile.id is None:
            continue
        relevant = groups.get(tile.id, set())
        pattern = "Unknown"
        if tile.features is not None:
            pattern = PatternClassifier.classify(tile.features).value.title()

        record = {
            "query_path": tile.file_path.replace("\\", "/"),
            "relevant_ids": sorted(relevant),
            "category": pattern,
            "query_id": tile.id,
        }
        lines.append(json.dumps(record))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate eval manifest JSONL.")
    parser.add_argument(
        "--output",
        default="eval/my_queries.jsonl",
        help="Output manifest path (default: eval/my_queries.jsonl).",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.72,
        help="Embedding cosine threshold for auto grouping (when --no-curated).",
    )
    parser.add_argument(
        "--no-curated",
        action="store_true",
        help="Use embedding similarity instead of curated visual groups.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    count = generate_manifest(
        output,
        min_similarity=args.min_similarity,
        use_curated=not args.no_curated,
    )
    print(f"Wrote {count} queries to {output.resolve()}")
    print("Run: python dev_tools/eval_recall_precision.py --manifest eval/my_queries.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
