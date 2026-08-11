#!/usr/bin/env python3
"""
Explain why a query ranked catalog candidates the way production search does.

Usage:
  python scripts/explain_search.py path/to/query.jpg --catalog path/to/index --top 10
  python scripts/explain_search.py path/to/query.jpg --catalog path/to/index \\
      --find-tile PGYS2319 --top 30 --pool-size 100
  python scripts/explain_search.py --show-index-crop path/to/PGYS2319.jpg \\
      --output-dir /tmp/index_crop_debug
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.candidate_filter import CandidateFilter
from src.ai.embedder import DINOv2Embedder
from src.ai.feature_extractor import FeatureExtractor
from src.ai.debug.index_crop_debug import (
    IndexCropReport,
    format_index_crop_report,
    show_index_crops,
)
from src.ai.pattern_classifier import PatternClassifier
from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.reranker import HybridReRanker
from src.ai.search_quality.query_analyzer import analyze_query
from src.ai.vector_index import FaissIndexManager
from src.config.settings import AppSettings
from src.core.models import TileImage
from src.core.use_cases.search_tiles import (
    ORB_BOOST_MAX,
    ORB_MAX_CANDIDATES,
    ORB_VERIFICATION_BAND,
    SearchTilesUseCase,
    _QUERY_SELF_MATCH_SCORE,
    _WEAK_RESULT_ABSOLUTE_RAW_FLOOR,
    _WEAK_RESULT_RELATIVE_FLOOR,
)
from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteImageRepository
from src.utils.image_utils import compute_dhash, compute_sha256


@dataclass(frozen=True, slots=True)
class CandidateExplain:
    tile_id: int
    file_name: str
    rank: int | None
    final_score: float
    embedding: float
    color: float
    texture: float
    edge: float
    pattern: float
    pattern_compat: float
    color_penalty: float
    faiss_cos: float
    winning_view_index: int | None
    orb_nudge: float
    exact_match: bool
    weak_filter_kept: bool
    weak_filter_dropped: bool


@dataclass(frozen=True, slots=True)
class ExplainReport:
    query_path: str
    query_kind: str
    query_view_count: int
    weak_filter_min_raw: float
    weak_filter_top1_cleared: bool
    candidates: tuple[CandidateExplain, ...]
    pool_size: int
    unique_ids_in_pool: int


@dataclass(frozen=True, slots=True)
class TileLookupResult:
    """Where a specific catalog tile landed in the production search path."""

    tile_id: int
    file_name: str
    file_path: str
    in_faiss_pool: bool
    faiss_pool_rank: int | None
    faiss_cos: float | None
    winning_view_index: int | None
    rerank_rank: int | None
    final_score: float | None
    embedding: float | None
    color: float | None
    texture: float | None
    edge: float | None
    pattern: float | None
    weak_filter_kept: bool
    weak_filter_dropped: bool
    weak_filter_min_raw: float


def resolve_catalog_paths(catalog: Path) -> tuple[str, str, str]:
    catalog = catalog.expanduser().resolve()
    if not catalog.is_dir():
        raise FileNotFoundError(f"Catalog directory not found: {catalog}")

    candidates = [
        (
            catalog / "database" / "tiles.db",
            catalog / "index" / "tiles.index",
            catalog / "thumbnails",
        ),
        (
            catalog / "tiles.db",
            catalog / "tiles.index",
            catalog / "thumbnails",
        ),
    ]
    for db_path, index_path, thumb_path in candidates:
        if db_path.is_file() and index_path.is_file():
            return str(db_path), str(index_path), str(thumb_path)

    raise FileNotFoundError(
        f"Could not find tiles.db and tiles.index under {catalog}. "
        "Expected database/tiles.db + index/tiles.index or flat tiles.db + tiles.index."
    )


def resolve_tile_reference(
    repo: SQLiteImageRepository,
    ref: str,
) -> TileImage | None:
    """Resolve a tile by numeric id, filename stem, or absolute catalog path."""
    ref = ref.strip()
    if ref.isdigit():
        return repo.get_by_id(int(ref))
    path = Path(ref)
    if path.is_file():
        tile = repo.get_by_path(str(path.resolve()))
        if tile is not None:
            return tile
    lookup = getattr(repo, "get_indexed_by_file_stem", None)
    if callable(lookup):
        return lookup(ref)
    stem = ref.lower()
    for tile in repo.get_all():
        if tile.is_indexed and Path(tile.file_name).stem.lower() == stem:
            return tile
    return None


def bootstrap_search(
    catalog: Path,
    *,
    enable_orb: bool = True,
) -> SearchTilesUseCase:
    database_path, index_path, thumbnail_dir = resolve_catalog_paths(catalog)
    print(
        f"[explain_search] catalog paths:\n"
        f"  database={database_path}\n"
        f"  index={index_path}\n"
        f"  thumbnails={thumbnail_dir}",
        flush=True,
    )
    db_context = DatabaseContext(db_path=database_path)
    repo = SQLiteImageRepository(db_context=db_context)
    feature_extractor = FeatureExtractor(embedder=DINOv2Embedder())
    vector_index = FaissIndexManager(index_path=index_path, dimension=1024)
    feature_extractor.load_model()
    vector_index.load_index()
    return SearchTilesUseCase(
        image_repository=repo,
        feature_extractor=feature_extractor,
        vector_index=vector_index,
        thumbnail_dir=thumbnail_dir,
        enable_orb_verification=enable_orb,
    )


def _compute_weak_filter_threshold(
    reranked: list[tuple[float, TileImage, bool]],
) -> float:
    if not reranked:
        return _WEAK_RESULT_ABSOLUTE_RAW_FLOOR
    reference_score = reranked[0][0]
    if reranked[0][2]:
        for score, _, exact_match in reranked[1:]:
            if not exact_match:
                reference_score = score
                break
    return max(
        reference_score * _WEAK_RESULT_RELATIVE_FLOOR,
        _WEAK_RESULT_ABSOLUTE_RAW_FLOOR,
    )


def _score_pair(
    reranker: HybridReRanker,
    query_features,
    tile: TileImage,
    query_pattern_type,
) -> tuple[Any, float, float]:
    candidate_pattern_type = PatternClassifier.classify(tile.features)
    hybrid = reranker.score(
        query_features,
        tile.features,
        query_pattern_type=query_pattern_type,
        candidate_pattern_type=candidate_pattern_type,
    )
    compat = PatternClassifier.compatibility_adjustment(
        query_pattern_type,
        candidate_pattern_type,
    )
    color_penalty = CandidateFilter.dominant_color_penalty(
        query_features,
        tile.features,
    )
    return hybrid, compat, color_penalty


def explain_search(
    use_case: SearchTilesUseCase,
    query_image_path: str | Path,
    *,
    top_k: int = 10,
    pool_size: int | None = None,
) -> tuple[
    ExplainReport,
    list[tuple[float, TileImage, bool, float, int | None, Any, float, float]],
    dict[str, Any],
]:
    query_path = Path(query_image_path)
    if not query_path.is_file():
        raise FileNotFoundError(f"Query image not found: {query_path}")

    from PIL import Image

    preloaded = Image.open(query_path)
    analysis = analyze_query(preloaded.convert("RGB"))

    query_features, query_embeddings = use_case._feature_extractor.extract_for_search(
        str(query_path),
        preloaded=preloaded,
    )
    query_pattern_type = PatternClassifier.classify(query_features)
    query_sha256 = compute_sha256(query_path)
    query_dhash = compute_dhash(query_path)

    total_vectors = use_case._index.get_total_count()
    if pool_size is not None:
        search_k = min(max(int(pool_size), 1), total_vectors)
    else:
        search_k = SearchTilesUseCase._compute_faiss_search_k(
            top_k,
            total_vectors,
        )
    matching_ids, faiss_scores, faiss_winning_views = use_case._search_faiss_multi_crop(
        query_embeddings or [query_features.embedding],
        search_k,
    )
    matched_tiles = use_case._repo.get_by_ids(matching_ids)
    tile_map = {tile.id: tile for tile in matched_tiles if tile.id is not None}

    reranker = use_case._reranker
    reranked: list[
        tuple[float, TileImage, bool, float, int | None, Any, float, float]
    ] = []

    for record_id in matching_ids:
        tile = tile_map.get(record_id)
        if tile is None or tile.features is None:
            continue

        hybrid, compat, color_penalty = _score_pair(
            reranker,
            query_features,
            tile,
            query_pattern_type,
        )
        exact_match = SearchTilesUseCase._is_exact_match(
            tile,
            query_sha256,
            query_dhash,
        )
        same_query_file = False
        try:
            same_query_file = Path(tile.file_path).resolve() == query_path.resolve()
        except OSError:
            same_query_file = False
        # Mirror SearchTilesUseCase.execute: demote same-file self-hit score but
        # keep exact_match=True so weak-filter reference uses the next peer.
        if exact_match and same_query_file:
            final_score = _QUERY_SELF_MATCH_SCORE
        elif exact_match:
            final_score = 1.0
        else:
            final_score = hybrid.final

        reranked.append(
            (
                final_score,
                tile,
                exact_match,
                float(faiss_scores.get(record_id, 0.0) or 0.0),
                faiss_winning_views.get(record_id),
                hybrid,
                compat,
                color_penalty,
            )
        )

    reranked.sort(key=lambda item: item[0], reverse=True)

    orb_nudges: dict[int, float] = {}
    if use_case._enable_orb_verification and use_case._orb_verifier is not None and reranked:
        query_gray = use_case._resolve_query_gray(query_path, preloaded)
        band = [
            (score, tile, exact)
            for score, tile, exact, *_rest in reranked[:ORB_MAX_CANDIDATES]
        ]
        if band:
            top_score = band[0][0]
            for idx, (score, tile, exact) in enumerate(band):
                if exact or (top_score - score) > ORB_VERIFICATION_BAND:
                    continue
                try:
                    from src.ai.preprocess.image_preprocessor import ImagePreprocessor

                    cand_gray = ImagePreprocessor.load(Path(tile.file_path)).convert("L")
                    cand_gray = np.asarray(cand_gray)
                    orb_score = float(
                        use_case._orb_verifier.score(query_gray, cand_gray)
                    )
                except OSError:
                    orb_score = 0.0
                if orb_score <= 0.0:
                    continue
                nudge = ORB_BOOST_MAX * orb_score
                orb_nudges[tile.id] = nudge
                boosted = min(1.0, float(score) + nudge)
                reranked[idx] = (
                    boosted,
                    tile,
                    exact,
                    reranked[idx][3],
                    reranked[idx][4],
                    reranked[idx][5],
                    reranked[idx][6],
                    reranked[idx][7],
                )
            reranked.sort(key=lambda item: item[0], reverse=True)

    simple = [(score, tile, exact) for score, tile, exact, *_ in reranked]
    min_raw = _compute_weak_filter_threshold(simple)
    kept_tuples = SearchTilesUseCase._filter_weak_results(simple, top_k)
    kept_ids = {tile.id for _, tile, _ in kept_tuples if tile.id is not None}

    rank_by_id = {
        tile.id: idx + 1
        for idx, (_, tile, _) in enumerate(kept_tuples)
        if tile.id is not None
    }

    candidates: list[CandidateExplain] = []
    for row in reranked[: top_k + 5]:
        score, tile, exact, faiss_cos, view_idx, hybrid, compat, color_penalty = row
        if tile.id is None:
            continue
        kept = tile.id in kept_ids
        dropped = not kept and score < min_raw and not exact
        candidates.append(
            CandidateExplain(
                tile_id=int(tile.id),
                file_name=tile.file_name,
                rank=rank_by_id.get(tile.id),
                final_score=float(score),
                embedding=float(hybrid.embedding),
                color=float(hybrid.color),
                texture=float(hybrid.texture),
                edge=float(hybrid.edge),
                pattern=float(hybrid.pattern),
                pattern_compat=float(compat),
                color_penalty=float(color_penalty),
                faiss_cos=float(faiss_cos),
                winning_view_index=view_idx,
                orb_nudge=float(orb_nudges.get(tile.id, 0.0)),
                exact_match=bool(exact),
                weak_filter_kept=kept,
                weak_filter_dropped=dropped,
            )
        )

    top1_faiss = float(faiss_scores.get(matching_ids[0], 0.0)) if matching_ids else 0.0
    top1_cleared = top1_faiss >= min_raw or (bool(reranked) and reranked[0][2])

    report = ExplainReport(
        query_path=str(query_path),
        query_kind=analysis.kind.value,
        query_view_count=len(query_embeddings or [query_features.embedding]),
        weak_filter_min_raw=float(min_raw),
        weak_filter_top1_cleared=bool(top1_cleared),
        candidates=tuple(candidates),
        pool_size=int(search_k),
        unique_ids_in_pool=len(matching_ids),
    )
    search_context = {
        "matching_ids": matching_ids,
        "faiss_scores": faiss_scores,
        "faiss_winning_views": faiss_winning_views,
        "kept_ids": kept_ids,
    }
    return report, reranked, search_context


def lookup_tile_in_search(
    reranked: list[
        tuple[float, TileImage, bool, float, int | None, Any, float, float]
    ],
    *,
    tile: TileImage,
    matching_ids: list[int],
    faiss_scores: dict[int, float],
    faiss_winning_views: dict[int, int],
    min_raw: float,
    kept_ids: set[int],
) -> TileLookupResult:
    tile_id = tile.id
    assert tile_id is not None

    faiss_rank = None
    if tile_id in matching_ids:
        faiss_rank = matching_ids.index(tile_id) + 1

    rerank_rank = None
    final_score = None
    embedding = None
    color = None
    texture = None
    edge = None
    pattern = None
    for idx, (score, t, exact, faiss_cos, view_idx, hybrid, _compat, _pen) in enumerate(
        reranked, start=1
    ):
        if t.id == tile_id:
            rerank_rank = idx
            final_score = float(score)
            embedding = float(hybrid.embedding)
            color = float(hybrid.color)
            texture = float(hybrid.texture)
            edge = float(hybrid.edge)
            pattern = float(hybrid.pattern)
            break

    kept = tile_id in kept_ids
    dropped = (
        not kept
        and final_score is not None
        and final_score < min_raw
        and rerank_rank is not None
    )

    return TileLookupResult(
        tile_id=int(tile_id),
        file_name=tile.file_name,
        file_path=tile.file_path,
        in_faiss_pool=faiss_rank is not None,
        faiss_pool_rank=faiss_rank,
        faiss_cos=faiss_scores.get(tile_id),
        winning_view_index=faiss_winning_views.get(tile_id),
        rerank_rank=rerank_rank,
        final_score=final_score,
        embedding=embedding,
        color=color,
        texture=texture,
        edge=edge,
        pattern=pattern,
        weak_filter_kept=kept,
        weak_filter_dropped=dropped,
        weak_filter_min_raw=float(min_raw),
    )


def format_tile_lookup(result: TileLookupResult) -> str:
    lines = [
        f"Tile lookup: {result.file_name} (id={result.tile_id})",
        f"  path: {result.file_path}",
    ]
    if not result.in_faiss_pool:
        lines.append(
            "  FAISS pool: ABSENT — never entered retrieval pool at this pool_size"
        )
        lines.append(
            "  → Root cause is index-time or pool too small; widen --pool-size "
            "or run --show-index-crop on the catalog sheet."
        )
        return "\n".join(lines)

    lines.extend(
        [
            f"  FAISS pool rank: {result.faiss_pool_rank}  "
            f"cos={result.faiss_cos:.3f}  view={result.winning_view_index}",
        ]
    )
    if result.rerank_rank is None:
        lines.append("  Hybrid rerank: not scored (missing features in DB?)")
    else:
        lines.append(
            f"  Hybrid rerank rank: {result.rerank_rank}  "
            f"final={result.final_score:.3f}"
        )
        lines.append(
            f"  Components: emb={result.embedding:.3f} color={result.color:.3f} "
            f"tex={result.texture:.3f} edge={result.edge:.3f} "
            f"pat={result.pattern:.3f}"
        )
        if result.weak_filter_kept:
            lines.append(
                f"  Weak filter: KEPT (floor={result.weak_filter_min_raw:.3f})"
            )
        elif result.weak_filter_dropped:
            lines.append(
                f"  Weak filter: DROPPED final={result.final_score:.3f} < "
                f"floor={result.weak_filter_min_raw:.3f}"
            )
        else:
            lines.append(
                f"  Weak filter: below top-K but above floor "
                f"(final={result.final_score:.3f}, floor={result.weak_filter_min_raw:.3f})"
            )
    return "\n".join(lines)


def format_report(report: ExplainReport) -> str:
    lines = [
        f"Query: {report.query_path}",
        f"Query kind: {report.query_kind}",
        f"Query views embedded: {report.query_view_count}",
        (
            f"FAISS pool: search_k={report.pool_size} "
            f"unique_ids={report.unique_ids_in_pool}"
        ),
        (
            f"Weak filter floor: min_raw={report.weak_filter_min_raw:.3f} "
            f"(relative={_WEAK_RESULT_RELATIVE_FLOOR}, "
            f"absolute={_WEAK_RESULT_ABSOLUTE_RAW_FLOOR})"
        ),
        f"Top-1 raw FAISS cleared weak floor: {report.weak_filter_top1_cleared}",
        "",
        (
            "rank  tile_id  final   emb    color  tex    edge   pat    "
            "compat  dom_pen  faiss   view  orb   kept  dropped  file"
        ),
    ]
    for row in report.candidates:
        rank_s = str(row.rank) if row.rank is not None else "-"
        view_s = (
            str(row.winning_view_index)
            if row.winning_view_index is not None
            else "-"
        )
        lines.append(
            f"{rank_s:>4}  {row.tile_id:<7}  {row.final_score:.3f}  "
            f"{row.embedding:.3f}  {row.color:.3f}  {row.texture:.3f}  "
            f"{row.edge:.3f}  {row.pattern:.3f}  {row.pattern_compat:+.3f}  "
            f"{row.color_penalty:+.3f}  {row.faiss_cos:.3f}  {view_s:>4}  "
            f"{row.orb_nudge:+.3f}  {str(row.weak_filter_kept):<5}  "
            f"{str(row.weak_filter_dropped):<7}  {row.file_name}"
        )
    dropped = [r for r in report.candidates if r.weak_filter_dropped]
    if dropped:
        lines.append("")
        lines.append("Weak-filter drops (shown above with dropped=True):")
        for row in dropped:
            lines.append(
                f"  tile_id={row.tile_id} final={row.final_score:.3f} "
                f"faiss={row.faiss_cos:.3f} file={row.file_name}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query",
        type=Path,
        nargs="?",
        default=None,
        help="Query image path (optional when using --show-index-crop alone)",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Catalog directory (database + FAISS index). Defaults to AppSettings.",
    )
    parser.add_argument("--top", type=int, default=10, help="Top-K candidates to show")
    parser.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="Override FAISS search_k (default: production formula, typically 100)",
    )
    parser.add_argument(
        "--find-tile",
        type=str,
        default=None,
        metavar="STEM_OR_ID",
        help="Report rank/scores for a specific catalog tile (e.g. PGYS2319)",
    )
    parser.add_argument(
        "--show-index-crop",
        type=str,
        default=None,
        metavar="PATH_OR_TILE",
        help="Save index-time crop PNGs for a catalog sheet (debug only)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/index_crop_debug"),
        help="Directory for --show-index-crop PNG output",
    )
    parser.add_argument(
        "--no-orb",
        action="store_true",
        help="Disable ORB near-tie verification",
    )
    parser.add_argument(
        "--parity-out",
        type=Path,
        default=None,
        help="Write JSON snapshot for cross-platform parity diff (Task 4)",
    )
    args = parser.parse_args(argv)

    if args.query is None and args.show_index_crop is None:
        parser.error("Provide a query image or --show-index-crop")

    if args.catalog is None:
        settings = AppSettings()
        catalog = Path(settings.database_path).parent.parent
    else:
        catalog = args.catalog

    exit_code = 0
    outputs: list[str] = []

    try:
        use_case: SearchTilesUseCase | None = None
        if args.query is not None or args.find_tile is not None:
            use_case = bootstrap_search(catalog, enable_orb=not args.no_orb)

        if args.show_index_crop is not None:
            crop_path = Path(args.show_index_crop)
            fx: FeatureExtractor | None = None
            if use_case is not None:
                fx = use_case._feature_extractor

            if crop_path.is_file():
                resolved_crop = crop_path
            elif use_case is not None:
                tile = resolve_tile_reference(use_case._repo, args.show_index_crop)
                if tile is None:
                    raise FileNotFoundError(
                        f"Tile not found in catalog: {args.show_index_crop}"
                    )
                resolved_crop = Path(tile.file_path)
            else:
                raise FileNotFoundError(
                    f"Cannot resolve --show-index-crop {args.show_index_crop!r}: "
                    "pass --catalog or an absolute image path"
                )

            if fx is None:
                fx = FeatureExtractor(embedder=DINOv2Embedder())
                try:
                    fx.load_model()
                except Exception:
                    fx = None

            crop_report = show_index_crops(
                resolved_crop,
                output_dir=args.output_dir,
                feature_extractor=fx,
            )
            outputs.append(format_index_crop_report(crop_report))

        if args.query is not None:
            assert use_case is not None
            report, reranked, search_context = explain_search(
                use_case,
                args.query,
                top_k=args.top,
                pool_size=args.pool_size,
            )
            outputs.append(format_report(report))

            lookup: TileLookupResult | None = None
            if args.find_tile is not None:
                tile = resolve_tile_reference(use_case._repo, args.find_tile)
                if tile is None:
                    raise FileNotFoundError(
                        f"Tile not found in catalog index: {args.find_tile}"
                    )
                lookup = lookup_tile_in_search(
                    reranked,
                    tile=tile,
                    matching_ids=search_context["matching_ids"],
                    faiss_scores=search_context["faiss_scores"],
                    faiss_winning_views=search_context["faiss_winning_views"],
                    min_raw=report.weak_filter_min_raw,
                    kept_ids=search_context["kept_ids"],
                )
                outputs.append(format_tile_lookup(lookup))

            if args.parity_out is not None:
                payload = {
                    "platform": {
                        "system": platform.system(),
                        "machine": platform.machine(),
                        "processor": platform.processor(),
                        "python": platform.python_version(),
                    },
                    "query": str(args.query),
                    "catalog": str(catalog),
                    "top_k": args.top,
                    "pool_size": report.pool_size,
                    "report": asdict(report),
                }
                if lookup is not None:
                    payload["tile_lookup"] = asdict(lookup)
                args.parity_out.parent.mkdir(parents=True, exist_ok=True)
                args.parity_out.write_text(
                    json.dumps(payload, indent=2, default=str),
                    encoding="utf-8",
                )
                outputs.append(f"Parity snapshot written: {args.parity_out}")

    except (FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\n\n".join(outputs))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
