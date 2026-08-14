"""
Cross-platform descriptor parity check — run this on the SAME Windows machine
with the SAME real files the client used, to compare against the Linux
sandbox reproduction.

Usage:
    python windows_parity_check.py "C:\\Users\\HP\\Documents\\Tiles\\xx.jpg.jpeg" "C:\\Users\\HP\\Documents\\Tiles\\PGYS2319.jpg"

This uses the exact same production code path as the app (prepare_index_primary,
the same one FeatureExtractor.extract_index_vectors() calls), simulating the
catalog_stored comparison the UI does when the query is itself an indexed tile.

Run from the repo root, on branch pr45-v2 / PR #45 latest commit (ca612eb or
later), so the code matches what's actually installed.
"""

import hashlib
import sys
from pathlib import Path

from src.ai.debug.index_crop_debug import show_index_crops, _features_from_preprocessed
from src.ai.preprocess.index_primary import prepare_index_primary
from src.core.models import TileImage


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python windows_parity_check.py <query_image> <catalog_sheet_image>")
        sys.exit(1)

    query = Path(sys.argv[1]).resolve()
    sheet = Path(sys.argv[2]).resolve()

    print("=" * 70)
    print("FILE IDENTITY CHECK — compare these SHA256 values against the")
    print("Linux sandbox run to confirm you're testing the exact same bytes")
    print("=" * 70)
    print(f"  query file : {query}")
    print(f"    sha256   : {sha256_of(query)}")
    print(f"  sheet file : {sheet}")
    print(f"    sha256   : {sha256_of(sheet)}")
    print()

    print("Computing prepare_index_primary() for both files (production path)...")
    q_feats = _features_from_preprocessed(prepare_index_primary(query).primary)
    c_feats = _features_from_preprocessed(prepare_index_primary(sheet).primary)

    class _RealFileRepo:
        def get_by_path(self, file_path: str):
            p = Path(file_path).resolve()
            if p == query:
                return TileImage(
                    id=1, file_path=str(p), file_name=p.name,
                    file_size=query.stat().st_size, dimensions="unknown",
                    is_indexed=True, sha256_hash=sha256_of(query), features=q_feats,
                )
            if p == sheet:
                return TileImage(
                    id=2, file_path=str(p), file_name=p.name,
                    file_size=sheet.stat().st_size, dimensions="unknown",
                    is_indexed=True, sha256_hash=sha256_of(sheet), features=c_feats,
                )
            return None

    report = show_index_crops(
        sheet,
        output_dir=Path("./windows_parity_debug_out"),
        query_path=query,
        query_mode="catalog",
        catalog_repo=_RealFileRepo(),
    )

    print()
    print("=" * 70)
    print("RESULT — compare against:")
    print("  Linux sandbox : color=0.999 texture=0.818 edge=0.968 pattern=0.663")
    print("  Client's real production log : color=0.968 texture=0.910 edge=0.476 pattern=0.342")
    print("=" * 70)
    print(f"  mode    : {report.parity.mode}")
    print(f"  color   : {report.parity.color}")
    print(f"  texture : {report.parity.texture}")
    print(f"  edge    : {report.parity.edge}")
    print(f"  pattern : {report.parity.pattern}")
    print()
    print("If this matches the Linux sandbox numbers (not the client's real log),")
    print("that rules out a Windows/Linux platform divergence -- the remaining")
    print("difference is almost certainly the client's actual indexed DB having")
    print("stored features from a different pass, or different source file bytes")
    print("than what's being tested here. Check the SHA256 values above against")
    print("the client's real files to confirm.")
    print()
    print("If this DOES match the client's real log instead, that's a genuine")
    print("Windows-vs-Linux numeric divergence in the descriptor pipeline --")
    print("a real, separate bug worth its own investigation.")


if __name__ == "__main__":
    main()
