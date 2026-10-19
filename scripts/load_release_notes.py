#!/usr/bin/env python3
"""
Load customer-facing release notes from ``docs/release_notes/<version>.md``.

Used by ``generate_update_manifest.py`` (in-app update dialog) and the Build
workflow (GitHub Release body). Single source of truth per version.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES_DIR = ROOT / "docs" / "release_notes"


def notes_path_for(version: str) -> Path:
    ver = version.lstrip("v").strip()
    return NOTES_DIR / f"{ver}.md"


def load_release_notes(version: str, *, root: Path = ROOT) -> str:
    """
    Return plain-text notes for ``version``.

    Strips a leading markdown ``#`` title line so the in-app dialog does not
    show a duplicated heading (the dialog already says “What's new”).
    """
    ver = version.lstrip("v").strip()
    path = (root / "docs" / "release_notes" / f"{ver}.md")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing release notes: {path.relative_to(root)} "
            f"(create docs/release_notes/{ver}.md before tagging v{ver})"
        )
    text = path.read_text(encoding="utf-8").strip()
    # Drop the first H1 so callers can supply their own title.
    lines = text.splitlines()
    if lines and re.match(r"^#\s+", lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="App version, e.g. 1.2.43 or v1.2.43")
    parser.add_argument(
        "--path-only",
        action="store_true",
        help="Print the notes file path instead of its contents",
    )
    args = parser.parse_args()
    path = notes_path_for(args.version)
    if args.path_only:
        print(path)
        return 0
    try:
        sys.stdout.write(load_release_notes(args.version))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
