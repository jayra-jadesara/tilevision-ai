#!/usr/bin/env python3
"""
Decode stored SQLite histograms and compare to a fresh recompute.

Use after a real UI "Rebuild Search Index" to verify batch indexing wrote
the same descriptors as ``extract_index_vectors()`` / ``prepare_index_primary``.

  python scripts/decode_stored_histograms.py \\
      --catalog /path/to/catalog \\
      --index PGYS2319.jpg \\
      --query xx.jpg.jpeg

Prints STORED (SQLite blobs), FRESH recompute, and pairwise similarities.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.descriptors.color_descriptor import ColorDescriptor
from src.ai.descriptors.edge_descriptor import EdgeDescriptor
from src.ai.descriptors.pattern_descriptor import PatternDescriptor
from src.ai.descriptors.texture_descriptor import TextureDescriptor
from src.ai.feature_extractor import FeatureExtractor
from src.ai.models import TileFeatures
from src.utils.image_utils import compute_sha256


def _resolve_db_path(catalog: Path) -> Path:
    catalog = catalog.expanduser().resolve()
    for candidate in (
        catalog / "database" / "tiles.db",
        catalog / "tiles.db",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find tiles.db under {catalog}")


def _resolve_image(catalog: Path, name: str) -> Path:
    catalog = catalog.expanduser().resolve()
    p = Path(name)
    if p.is_file():
        return p.resolve()
    for candidate in catalog.rglob(name):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find image {name!r} under {catalog}")


def _open_repo(catalog: Path):
    from src.data.db_context import DatabaseContext
    from src.data.sqlite_repository import SQLiteImageRepository

    db = DatabaseContext(str(_resolve_db_path(catalog)))
    return SQLiteImageRepository(db)


def _feature_version(repo, tile_id: int) -> int | None:
    try:
        with repo._db.session() as conn:
            row = conn.execute(
                "SELECT feature_version FROM tiles WHERE id = ?",
                (tile_id,),
            ).fetchone()
            return int(row[0]) if row else None
    except Exception:
        return None


def _lookup_stored(repo, image_path: Path) -> tuple | None:
    tile = repo.get_by_path(str(image_path.resolve()))
    if tile is None or tile.features is None:
        return None
    sha = compute_sha256(image_path)
    if tile.sha256_hash != sha:
        return None
    return tile, tile.features


def _sims(a: TileFeatures, b: TileFeatures) -> tuple[float, float, float, float]:
    return (
        float(ColorDescriptor.similarity(a.color_histogram, b.color_histogram)),
        float(TextureDescriptor.similarity(a.texture_histogram, b.texture_histogram)),
        float(EdgeDescriptor.similarity(a.edge_histogram, b.edge_histogram)),
        float(PatternDescriptor.similarity(a.pattern_features, b.pattern_features)),
    )


def _fresh_features(image_path: Path, extractor: FeatureExtractor) -> TileFeatures:
    features, _aux = extractor.extract_index_vectors(str(image_path))
    return features


def _format_sims(label: str, sims: tuple[float, float, float, float]) -> str:
    color, texture, edge, pattern = sims
    return (
        f"{label}: color={color:.3f} texture={texture:.3f} "
        f"edge={edge:.3f} pattern={pattern:.3f}"
    )


def _stored_matches_fresh(stored: TileFeatures, fresh: TileFeatures) -> bool:
    """SQLite stores histograms as float16 blobs."""

    def _rt(hist: np.ndarray) -> np.ndarray:
        return hist.astype(np.float16).astype(np.float32)

    return (
        np.allclose(_rt(stored.color_histogram), _rt(fresh.color_histogram))
        and np.allclose(_rt(stored.texture_histogram), _rt(fresh.texture_histogram))
        and np.allclose(_rt(stored.edge_histogram), _rt(fresh.edge_histogram))
        and np.allclose(_rt(stored.pattern_features), _rt(fresh.pattern_features))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="Catalog profile directory (contains database/tiles.db)",
    )
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help="Index/catalog sheet filename or path (e.g. PGYS2319.jpg)",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query filename or path (e.g. xx.jpg.jpeg)",
    )
    parser.add_argument(
        "--no-fresh",
        action="store_true",
        help="Skip DINOv2 fresh recompute (descriptor-only via prepare_index_primary)",
    )
    args = parser.parse_args(argv)

    try:
        repo = _open_repo(args.catalog)
        index_path = _resolve_image(args.catalog, args.index)
        query_path = _resolve_image(args.catalog, args.query)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    index_row = _lookup_stored(repo, index_path)
    query_row = _lookup_stored(repo, query_path)
    if index_row is None:
        print(f"ERROR: no stored features for index file {index_path}", file=sys.stderr)
        return 1
    if query_row is None:
        print(f"ERROR: no stored features for query file {query_path}", file=sys.stderr)
        return 1

    index_tile, index_stored = index_row
    query_tile, query_stored = query_row

    print(f"Index: {index_path.name}")
    print(
        f"  id={index_tile.id} feature_version={_feature_version(repo, index_tile.id)} "
        f"updated={index_tile.updated_time} sha256={index_tile.sha256_hash[:16]}..."
    )
    print(f"Query: {query_path.name}")
    print(
        f"  id={query_tile.id} feature_version={_feature_version(repo, query_tile.id)} "
        f"updated={query_tile.updated_time} sha256={query_tile.sha256_hash[:16]}..."
    )
    print()

    stored_sims = _sims(query_stored, index_stored)
    print(_format_sims("STORED (real DB)", stored_sims))

    if args.no_fresh:
        return 0

    extractor = FeatureExtractor()
    extractor.load_model()
    query_fresh = _fresh_features(query_path, extractor)
    index_fresh = _fresh_features(index_path, extractor)
    fresh_sims = _sims(query_fresh, index_fresh)
    print(_format_sims("FRESH recompute", fresh_sims))
    print()

    q_match = _stored_matches_fresh(query_stored, query_fresh)
    i_match = _stored_matches_fresh(index_stored, index_fresh)
    print(f"Query stored vs fresh match: {q_match}")
    print(f"Index stored vs fresh match: {i_match}")
    if q_match and i_match:
        print("PASS: stored histograms match fresh recompute.")
    else:
        print("FAIL: stored histograms diverge from fresh recompute — reindex may be stale or batch path still wrong.")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
