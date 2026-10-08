#!/usr/bin/env python3
"""
Component-level diagnosis for Auto / Precise / Manual crop searches.

Temp crop files under ``%TEMP%/tilevision_crops`` (or ``/tmp/tilevision_crops``)
are unique-named; ``last_autocrop.jpg`` / ``last_precise.jpg`` / ``last_manual.jpg``
are the stable copies written after each crop. This script:

1. Copies those ``last_*.jpg`` files into an output folder (so they survive).
2. Runs ``explain_search``-equivalent component breakdown for each against
   a catalog, optionally ``--find-tile PGYS2319``.
3. Also compares a fresh drop-search of ``--source`` (if given) and reports
   whether that path would hit the **catalog feature cache** (the usual reason
   drag-and-drop scores look much higher than crop-tool scores on the same photo).

Usage (Windows PowerShell example)::

  python scripts/diagnose_crop_tool_search.py `
    --catalog "$env:USERPROFILE\\.tilevision_ai" `
    --source "C:\\Users\\HP\\Documents\\Tiles\\xx.jpg.jpeg" `
    --find-tile PGYS2319 `
    --out C:\\Temp\\crop_diag

Then paste the printed emb/color/tex/edge/pat table back to the investigation.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _crops_dir() -> Path:
    return Path(tempfile.gettempdir()) / "tilevision_crops"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="Catalog profile (parent of database/ + index/)")
    parser.add_argument("--source", default=None, help="Original query photo (e.g. xx.jpg.jpeg)")
    parser.add_argument("--find-tile", default="PGYS2319")
    parser.add_argument("--out", type=Path, default=Path("crop_tool_diag_out"))
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    crops = _crops_dir()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"tilevision_crops dir: {crops}")
    if not crops.is_dir():
        print("ERROR: no tilevision_crops folder yet — run Auto/Precise/Manual crop once in the app.")
        return 2

    copied: list[tuple[str, Path]] = []
    for name in ("last_autocrop.jpg", "last_precise.jpg", "last_manual.jpg"):
        src = crops / name
        if src.is_file():
            dest = args.out / name
            shutil.copy2(src, dest)
            copied.append((name, dest))
            print(f"copied {src} → {dest} ({dest.stat().st_size} bytes)")
        else:
            print(f"missing {src}")

    if not copied and args.source is None:
        print("Nothing to diagnose.")
        return 2

    # Prefer explain_search CLI for production-parity hybrid components.
    import subprocess

    def run_explain(query: Path, tag: str) -> None:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "explain_search.py"),
            str(query),
            "--catalog",
            str(args.catalog),
            "--top",
            str(args.top),
            "--find-tile",
            args.find_tile,
            "--query-origin",
            "crop_tool" if "last_" in query.name or "tilevision_crops" in query.as_posix() else "auto",
            "--parity-out",
            str(args.out / f"{tag}_parity.json"),
        ]
        print("\n===", tag, "===")
        print(" ", " ".join(cmd))
        subprocess.run(cmd, check=False)

    if args.source:
        src = Path(args.source)
        if src.is_file():
            run_explain(src, "drop_source")
        else:
            print(f"WARNING: --source not found: {src}")

    for name, dest in copied:
        run_explain(dest, Path(name).stem)

    print(
        "\nNote: if drop_source logs 'Reusing indexed features for catalog query', "
        "its scores are catalog-cache (index-time features), not a fresh embed of "
        "the same pixels the crop tools search. Compare crop rows to each other "
        "and to a fresh extract — not naively to a cache hit."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
