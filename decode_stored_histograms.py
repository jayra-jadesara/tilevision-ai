"""
DECISIVE TEST: decode the literal stored edge/pattern histogram bytes for
these two tiles from the real database, and score them with the exact same
EdgeDescriptor/PatternDescriptor.similarity() functions the app uses.

This settles the last open question:
  - If this reproduces ~0.476/0.342 (matching the client's real production
    log): the STORED bytes genuinely differ from anything a fresh
    prepare_index_primary() recompute produces (~0.968/0.663, confirmed
    identical on both Linux and Windows). That means the real batch
    "Rebuild Search Index" pipeline computes something different from the
    single-file prepare_index_primary() path used by all our debug tools
    so far -- a real, still-unfound divergence between batch indexing and
    the debug/diagnostic code path.
  - If this instead reproduces ~0.968/0.663 (matching fresh recompute): the
    stored data is fine, and explain_search.py itself has a bug in how it
    reads/reports the live component scores -- a bug in the diagnostic tool,
    not the data.

Usage:
    python decode_stored_histograms.py "C:\\Users\\HP\\.tilevision_ai"
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np

from src.ai.descriptors.edge_descriptor import EdgeDescriptor
from src.ai.descriptors.pattern_descriptor import PatternDescriptor
from src.ai.descriptors.color_descriptor import ColorDescriptor
from src.ai.descriptors.texture_descriptor import TextureDescriptor


def deserialize_histogram(blob: bytes) -> np.ndarray:
    """Matches SQLiteImageRepository._deserialize_histogram exactly."""
    return np.frombuffer(blob, dtype=np.float16).astype(np.float32)


def deserialize_vector(blob: bytes) -> np.ndarray:
    """Matches SQLiteImageRepository._deserialize_vector exactly."""
    return np.frombuffer(blob, dtype=np.float32).copy()


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python decode_stored_histograms.py <catalog_dir>")
        sys.exit(1)

    db_path = Path(sys.argv[1]) / "database" / "tiles.db"
    print(f"Opening: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    rows = {}
    for name in ("xx.jpg.jpeg", "PGYS2319.jpg"):
        row = conn.execute(
            """
            SELECT color_histogram, texture_histogram, edge_histogram,
                   pattern_features
            FROM tiles WHERE file_name = ? COLLATE NOCASE
            """,
            (name,),
        ).fetchone()
        if row is None:
            print(f"ERROR: {name} not found")
            sys.exit(1)
        rows[name] = row

    q = rows["xx.jpg.jpeg"]
    c = rows["PGYS2319.jpg"]

    q_color = deserialize_histogram(q["color_histogram"])
    c_color = deserialize_histogram(c["color_histogram"])
    q_texture = deserialize_histogram(q["texture_histogram"])
    c_texture = deserialize_histogram(c["texture_histogram"])
    q_edge = deserialize_histogram(q["edge_histogram"])
    c_edge = deserialize_histogram(c["edge_histogram"])
    q_pattern = deserialize_vector(q["pattern_features"])
    c_pattern = deserialize_vector(c["pattern_features"])

    print()
    print("=" * 70)
    print("SCORING THE LITERAL STORED BYTES (decisive test)")
    print("=" * 70)
    print(f"  color   : {ColorDescriptor.similarity(q_color, c_color)}")
    print(f"  texture : {TextureDescriptor.similarity(q_texture, c_texture)}")
    print(f"  edge    : {EdgeDescriptor.similarity(q_edge, c_edge)}")
    print(f"  pattern : {PatternDescriptor.similarity(q_pattern, c_pattern)}")
    print()
    print("Compare against:")
    print("  Fresh recompute (Linux + Windows, confirmed identical):")
    print("    color=0.999 texture=0.818 edge=0.968 pattern=0.663")
    print("  explain_search.py live production output:")
    print("    color=0.968 texture=0.910 edge=0.476 pattern=0.342")
    print()
    print("If the numbers above match the FRESH recompute row, the stored")
    print("data is correct and explain_search.py has a reporting bug.")
    print("If they match the LIVE PRODUCTION row instead, the real batch")
    print("indexing pipeline genuinely computed different histograms than")
    print("prepare_index_primary() does when called standalone -- a real,")
    print("still-unfound divergence between batch indexing and every debug")
    print("tool used in this investigation so far.")


if __name__ == "__main__":
    main()
