"""
TileVision AI — Offline Recall@K / Precision@K evaluation.

Measures end-to-end search quality (FAISS + candidate filter + reranker)
against a ground-truth manifest or an auto-generated catalog heuristic.

Usage:
    # Explicit ground-truth manifest (JSONL, one object per line):
    python dev_tools/eval_recall_precision.py --manifest eval/queries.jsonl

    # Auto mode: each indexed tile is a query; relevant tiles share product_code
    python dev_tools/eval_recall_precision.py --catalog-auto --max-queries 50

    # Custom data paths:
    python dev_tools/eval_recall_precision.py --catalog-auto \\
        --database-path "C:/path/tiles.db" --index-path "C:/path/tiles.index"

Manifest format (JSONL):
    {"query_path": "D:/tiles/Brand_Floor_Grey_60x60_ABC.jpg",
     "relevant_ids": [12, 45, 67],
     "category": "Floor"}

IMPORTANT: Requires a fully re-indexed catalog (feature_version=4).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ai.embedder import DINOv2Embedder
from src.ai.feature_extractor import FeatureExtractor
from src.ai.pattern_classifier import PatternClassifier
from src.ai.vector_index import FaissIndexManager
from src.config.settings import AppSettings
from src.core.use_cases.search_tiles import SearchTilesUseCase
from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteImageRepository

logger = logging.getLogger("tilevision.eval")


@dataclass
class EvalQuery:
    query_path: str
    relevant_ids: Set[int]
    category: str = "Unknown"
    query_id: Optional[int] = None


@dataclass
class MetricAccumulator:
    recall_sums: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    precision_sums: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    query_count: int = 0


def _parse_k_values(raw: str) -> List[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or any(k < 1 for k in values):
        raise argparse.ArgumentTypeError("K values must be positive integers.")
    return values


def _load_manifest(path: Path) -> List[EvalQuery]:
    queries: List[EvalQuery] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            query_path = str(record["query_path"])
            relevant_ids = {int(tile_id) for tile_id in record["relevant_ids"]}
            queries.append(
                EvalQuery(
                    query_path=query_path,
                    relevant_ids=relevant_ids,
                    category=str(record.get("category", "Unknown")),
                    query_id=record.get("query_id"),
                )
            )
    if not queries:
        raise ValueError(f"No queries found in manifest: {path}")
    return queries


def _build_catalog_queries(
    repo: SQLiteImageRepository,
    *,
    max_queries: Optional[int],
    min_relevant: int,
) -> List[EvalQuery]:
    tiles = [tile for tile in repo.get_all() if tile.is_indexed and tile.id is not None]
    if not tiles:
        raise ValueError("No indexed tiles found in the catalog.")

    by_product: Dict[str, List[int]] = defaultdict(list)
    for tile in tiles:
        code = (tile.product_code or "").strip()
        if code and code.lower() != "unknown":
            by_product[code].append(tile.id)

    queries: List[EvalQuery] = []
    for tile in tiles:
        code = (tile.product_code or "").strip()
        if code and code.lower() != "unknown":
            relevant = {tile_id for tile_id in by_product[code] if tile_id != tile.id}
        else:
            relevant = {
                other.id
                for other in tiles
                if other.id != tile.id
                and other.category == tile.category
                and other.color == tile.color
                and other.category.lower() != "unknown"
            }

        if len(relevant) < min_relevant:
            continue

        queries.append(
            EvalQuery(
                query_path=tile.file_path,
                relevant_ids=relevant,
                category=tile.category or "Unknown",
                query_id=tile.id,
            )
        )

    if max_queries is not None:
        queries = queries[:max_queries]

    if not queries:
        raise ValueError(
            "No catalog queries met min_relevant threshold. "
            "Index more tiles or lower --min-relevant."
        )
    return queries


def _compute_metrics(
    retrieved_ids: Sequence[int],
    relevant_ids: Set[int],
    k_values: Sequence[int],
) -> tuple[Dict[int, float], Dict[int, float]]:
    recall: Dict[int, float] = {}
    precision: Dict[int, float] = {}
    relevant_count = len(relevant_ids)
    if relevant_count == 0:
        return {k: 0.0 for k in k_values}, {k: 0.0 for k in k_values}

    for k in k_values:
        top_k = retrieved_ids[:k]
        hits = len(set(top_k) & relevant_ids)
        recall[k] = hits / relevant_count
        precision[k] = hits / k
    return recall, precision


def _accumulate(
    bucket: MetricAccumulator,
    recall: Dict[int, float],
    precision: Dict[int, float],
    k_values: Sequence[int],
) -> None:
    bucket.query_count += 1
    for k in k_values:
        bucket.recall_sums[k] += recall[k]
        bucket.precision_sums[k] += precision[k]


def _format_table(
    rows: Dict[str, MetricAccumulator],
    k_values: Sequence[int],
) -> str:
    headers = ["group", "queries"]
    for k in k_values:
        headers.extend([f"R@{k}", f"P@{k}"])

    col_widths = [max(len(h), 10) for h in headers]
    lines = [" | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))]
    lines.append("-+-".join("-" * w for w in col_widths))

    for group in sorted(rows):
        acc = rows[group]
        if acc.query_count == 0:
            continue
        cells = [group.ljust(col_widths[0]), str(acc.query_count).rjust(col_widths[1])]
        idx = 2
        for k in k_values:
            recall_avg = acc.recall_sums[k] / acc.query_count
            precision_avg = acc.precision_sums[k] / acc.query_count
            cells.append(f"{recall_avg:.3f}".rjust(col_widths[idx]))
            idx += 1
            cells.append(f"{precision_avg:.3f}".rjust(col_widths[idx]))
            idx += 1
        lines.append(" | ".join(cells))

    return "\n".join(lines)


def _bootstrap_search(
    database_path: str,
    index_path: str,
    thumbnail_dir: str,
) -> tuple[SearchTilesUseCase, SQLiteImageRepository, FaissIndexManager]:
    db_context = DatabaseContext(db_path=database_path)
    repo = SQLiteImageRepository(db_context=db_context)
    feature_extractor = FeatureExtractor(embedder=DINOv2Embedder())
    vector_index = FaissIndexManager(index_path=index_path, dimension=1024)

    print("Loading DINOv2 model and FAISS index...")
    feature_extractor.load_model()
    vector_index.load_index()

    search = SearchTilesUseCase(
        image_repository=repo,
        feature_extractor=feature_extractor,
        vector_index=vector_index,
        thumbnail_dir=thumbnail_dir,
    )
    return search, repo, vector_index


def run_evaluation(args: argparse.Namespace) -> int:
    settings = AppSettings()
    database_path = args.database_path or settings.database_path
    index_path = args.index_path or settings.index_path
    thumbnail_dir = args.thumbnail_dir or settings.thumbnail_dir
    k_values = args.k_values

    search, repo, vector_index = _bootstrap_search(
        database_path=database_path,
        index_path=index_path,
        thumbnail_dir=thumbnail_dir,
    )

    version_status = repo.get_feature_version_status()
    if not version_status.is_compatible and version_status.stale_count > 0:
        print(
            "WARNING: Indexed features may be stale. "
            f"Indexed={version_status.indexed_count}, "
            f"stale={version_status.stale_count}. "
            f"{version_status.message} "
            "Re-index before trusting these numbers."
        )

    index_count = vector_index.get_total_count()
    print(f"FAISS vectors: {index_count}")

    if args.manifest:
        queries = _load_manifest(Path(args.manifest))
    else:
        queries = _build_catalog_queries(
            repo,
            max_queries=args.max_queries,
            min_relevant=args.min_relevant,
        )

    print(f"Running {len(queries)} queries at K={k_values} ...")

    tile_by_id = {
        tile.id: tile
        for tile in repo.get_all()
        if tile.id is not None
    }

    overall = MetricAccumulator()
    by_category: Dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    by_pattern: Dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)

    max_k = max(k_values)
    failures = 0
    t0 = time.perf_counter()

    for idx, query in enumerate(queries, start=1):
        query_file = Path(query.query_path)
        if not query_file.exists():
            logger.warning("Skipping missing query image: %s", query.query_path)
            failures += 1
            continue

        try:
            results = search.execute(str(query_file), top_k=max_k)
        except Exception as exc:
            logger.error("Query failed for %s: %s", query.query_path, exc)
            failures += 1
            continue

        retrieved_ids: List[int] = []
        for result in results:
            if result.tile.id is None:
                continue
            if query.query_id is not None and result.tile.id == query.query_id:
                continue
            retrieved_ids.append(result.tile.id)

        relevant_ids = set(query.relevant_ids)
        if query.query_id is not None:
            relevant_ids.discard(query.query_id)

        recall, precision = _compute_metrics(retrieved_ids, relevant_ids, k_values)

        _accumulate(overall, recall, precision, k_values)
        _accumulate(by_category[query.category], recall, precision, k_values)

        pattern_label = "unknown"
        query_tile = tile_by_id.get(query.query_id) if query.query_id is not None else None
        if query_tile is None:
            query_tile = repo.get_by_path(str(query_file.resolve()))
        if query_tile and query_tile.features is not None:
            pattern_label = PatternClassifier.classify(query_tile.features).value

        _accumulate(by_pattern[pattern_label], recall, precision, k_values)

        if idx % 10 == 0 or idx == len(queries):
            elapsed = time.perf_counter() - t0
            print(f"  [{idx}/{len(queries)}] elapsed {elapsed:.1f}s")

    elapsed = time.perf_counter() - t0
    print()
    print(f"Completed in {elapsed:.1f}s ({failures} skipped/failed)")
    print()
    print("=== Overall ===")
    print(_format_table({"ALL": overall}, k_values))
    print()
    print("=== By showroom category ===")
    print(_format_table(dict(by_category), k_values))
    print()
    print("=== By query pattern family ===")
    print(_format_table(dict(by_pattern), k_values))

    if args.output:
        payload = {
            "k_values": k_values,
            "query_count": overall.query_count,
            "failures": failures,
            "elapsed_seconds": round(elapsed, 2),
            "index_count": index_count,
            "feature_version_stale": (
                not version_status.is_compatible and version_status.stale_count > 0
            ),
            "overall": {
                str(k): {
                    "recall": overall.recall_sums[k] / max(overall.query_count, 1),
                    "precision": overall.precision_sums[k] / max(overall.query_count, 1),
                }
                for k in k_values
            },
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nWrote summary JSON to {out_path}")

    return 0 if overall.query_count > 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate TileVision search Recall@K and Precision@K.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--manifest",
        help="Path to JSONL ground-truth manifest.",
    )
    source.add_argument(
        "--catalog-auto",
        action="store_true",
        help="Auto-generate queries from indexed catalog (same product_code).",
    )

    parser.add_argument(
        "--k-values",
        type=_parse_k_values,
        default=_parse_k_values("1,5,10,20"),
        help="Comma-separated K values (default: 1,5,10,20).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit number of catalog-auto queries.",
    )
    parser.add_argument(
        "--min-relevant",
        type=int,
        default=1,
        help="Minimum relevant tiles required per catalog-auto query.",
    )
    parser.add_argument("--database-path", default=None)
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--thumbnail-dir", default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON summary.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    return run_evaluation(args)


if __name__ == "__main__":
    raise SystemExit(main())
