#!/usr/bin/env python3
"""
Explain why a query ranked catalog candidates the way production search does.

Usage:
  python scripts/explain_search.py path/to/query.jpg --catalog path/to/index --top 10
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.candidate_filter import CandidateFilter
from src.ai.embedder import DINOv2Embedder
from src.ai.feature_extractor import FeatureExtractor
from src.ai.pattern_classifier import PatternClassifier
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


def bootstrap_search(
    catalog: Path,
    *,
    enable_orb: bool = True,
) -> SearchTilesUseCase:
    database_path, index_path, thumbnail_dir = resolve_catalog_paths(catalog)
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
) -> ExplainReport:
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

    search_k = SearchTilesUseCase._compute_faiss_search_k(
        top_k,
        use_case._index.get_total_count(),
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
        final_score = 1.0 if exact_match else hybrid.final

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

    return ExplainReport(
        query_path=str(query_path),
        query_kind=analysis.kind.value,
        query_view_count=len(query_embeddings or [query_features.embedding]),
        weak_filter_min_raw=float(min_raw),
        weak_filter_top1_cleared=bool(top1_cleared),
        candidates=tuple(candidates),
    )


def format_report(report: ExplainReport) -> str:
    lines = [
        f"Query: {report.query_path}",
        f"Query kind: {report.query_kind}",
        f"Query views embedded: {report.query_view_count}",
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
    parser.add_argument("query", type=Path, help="Query image path")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Catalog directory (database + FAISS index). Defaults to AppSettings.",
    )
    parser.add_argument("--top", type=int, default=10, help="Top-K candidates to show")
    parser.add_argument(
        "--no-orb",
        action="store_true",
        help="Disable ORB near-tie verification",
    )
    args = parser.parse_args(argv)

    if args.catalog is None:
        settings = AppSettings()
        catalog = Path(settings.database_path).parent.parent
    else:
        catalog = args.catalog

    try:
        use_case = bootstrap_search(catalog, enable_orb=not args.no_orb)
        report = explain_search(use_case, args.query, top_k=args.top)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
