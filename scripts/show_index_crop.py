#!/usr/bin/env python3
"""
Save index-time crop PNGs for a catalog sheet (no DINOv2 / FAISS required).

Uses the same ``prepare_index_primary()`` path as production indexing.

  python scripts/show_index_crop.py path/to/PGYS2319.jpg --output-dir /tmp/index_crop_debug

To match explain_search hybrid color/texture/edge/pattern components:

  python scripts/show_index_crop.py path/to/PGYS2319.jpg \\
      --query path/to/xx.jpg --output-dir /tmp/index_crop_debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.debug.index_crop_debug import format_index_crop_report, show_index_crops


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
        help=(
            "Optional query image. Saves query preprocess letterbox and prints "
            "descriptor similarities vs index primary (production hybrid pair)."
        ),
    )
    args = parser.parse_args(argv)
    try:
        report = show_index_crops(
            args.image,
            output_dir=args.output_dir,
            query_path=args.query,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(format_index_crop_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
