"""
Inspect the ACTUAL stored feature_version and histogram data for two tiles
in the real catalog database -- raw sqlite3, no app imports needed, so this
can't break from an unrelated code refactor.

Usage:
    python inspect_stored_features.py "C:\\Users\\HP\\.tilevision_ai"
"""

import sqlite3
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python inspect_stored_features.py <catalog_dir>")
        sys.exit(1)

    catalog_dir = Path(sys.argv[1])
    db_path = catalog_dir / "database" / "tiles.db"
    print(f"Opening: {db_path}")
    if not db_path.exists():
        print("ERROR: file does not exist at that path.")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    for name in ("xx.jpg.jpeg", "PGYS2319.jpg"):
        rows = conn.execute(
            """
            SELECT id, file_path, file_name, sha256_hash, is_indexed,
                   feature_version, pattern_feature_version,
                   embedding_dimension, embedding_model,
                   length(color_histogram) as color_len,
                   length(texture_histogram) as texture_len,
                   length(edge_histogram) as edge_len,
                   length(pattern_features) as pattern_len,
                   updated_time
            FROM tiles
            WHERE file_name = ? COLLATE NOCASE
            """,
            (name,),
        ).fetchall()

        if not rows:
            print(f"\n{name}: NOT FOUND by filename in tiles table")
            continue

        for row in rows:
            print(f"\n=== {row['file_path']} (id={row['id']}) ===")
            print(f"  is_indexed              : {row['is_indexed']}")
            print(f"  sha256_hash             : {row['sha256_hash']}")
            print(f"  feature_version         : {row['feature_version']}")
            print(f"  pattern_feature_version : {row['pattern_feature_version']}")
            print(f"  embedding_model         : {row['embedding_model']}")
            print(f"  embedding_dimension     : {row['embedding_dimension']}")
            print(f"  color_histogram bytes   : {row['color_len']}")
            print(f"  texture_histogram bytes : {row['texture_len']}")
            print(f"  edge_histogram bytes    : {row['edge_len']}")
            print(f"  pattern_features bytes  : {row['pattern_len']}")
            print(f"  updated_time            : {row['updated_time']}")

    conn.close()
    print()
    print("=" * 70)
    print("Expected: feature_version should be 16 for BOTH rows (matching")
    print("CURRENT_FEATURE_VERSION after the v16 edge descriptor fix), and")
    print("updated_time for both should match the recent rebuild, not an")
    print("earlier date. If either shows an older feature_version or an")
    print("updated_time that doesn't match the rebuild, that tile's stored")
    print("descriptors are stale despite the UI reporting the rebuild as")
    print("complete -- a real bug in the rebuild/write path, not a code fix")
    print("problem.")
    print("=" * 70)


if __name__ == "__main__":
    main()
