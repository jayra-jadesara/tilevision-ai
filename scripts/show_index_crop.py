#!/usr/bin/env python3
"""
Save index-time crop PNGs for a catalog sheet (no DINOv2 / FAISS required).

Uses the same ``prepare_index_primary()`` path as production indexing.

  python scripts/show_index_crop.py path/to/PGYS2319.jpg --output-dir /tmp/index_crop_debug

Compare descriptors the way the UI does for an already-indexed query file
(catalog-cache / stored-vs-stored) AND show the ad-hoc fresh path:

  python scripts/show_index_crop.py path/to/PGYS2319.jpg \\
      --query path/to/xx.jpg --output-dir /tmp/index_crop_debug

With a real catalog DB, compare the exact stored SQLite blobs:

  python scripts/show_index_crop.py path/to/PGYS2319.jpg \\
      --query path/to/xx.jpg.jpeg --catalog /path/to/catalog \\
      --output-dir /tmp/index_crop_debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.debug.index_crop_debug import format_index_crop_report, show_index_crops


def _resolve_db_path(catalog: Path) -> Path:
    catalog = catalog.expanduser().resolve()
    for candidate in (
        catalog / "database" / "tiles.db",
        catalog / "tiles.db",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find tiles.db under {catalog}"
    )


def _open_catalog_repo(catalog: Path):
    from src.data.db_context import DatabaseContext
    from src.data.sqlite_repository import SQLiteImageRepository

    db = DatabaseContext(str(_resolve_db_path(catalog)))
    return SQLiteImageRepository(db)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Catalog sheet image path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/index_crop_debug"),
        help="Directory for saved PNG crops",
    )
    parser.add_argument(
        "--query",
        type=Path,
        default=None,
        help="Optional query image for descriptor parity",
    )
    parser.add_argument(
        "--query-mode",
        choices=("auto", "catalog", "fresh", "both"),
        default="auto",
        help=(
            "auto/both: catalog path primary + fresh alternate; "
            "catalog: UI catalog-cache simulation only; "
            "fresh: preprocess_for_query only"
        ),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help=(
            "Catalog profile directory. When set with --query, parity uses "
            "stored SQLite TileFeatures if the query path is an indexed tile."
        ),
    )
    args = parser.parse_args(argv)

    catalog_repo = None
    if args.catalog is not None:
        try:
            catalog_repo = _open_catalog_repo(args.catalog)
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"ERROR: cannot open catalog: {exc}", file=sys.stderr)
            return 1

    try:
        report = show_index_crops(
            args.image,
            output_dir=args.output_dir,
            query_path=args.query,
            query_mode=args.query_mode,
            catalog_repo=catalog_repo,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(format_index_crop_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
