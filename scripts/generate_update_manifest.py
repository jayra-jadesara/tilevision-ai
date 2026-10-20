#!/usr/bin/env python3
"""Generate update_manifest.json for GitHub Releases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow ``python scripts/generate_update_manifest.py`` without PYTHONPATH.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from load_release_notes import load_release_notes  # noqa: E402


def main() -> int:
    if len(sys.argv) < 5:
        print(
            "Usage: generate_update_manifest.py VERSION WINDOWS_URL "
            "MACOS_INTEL_URL MACOS_ARM64_URL [NOTES_OR_@FILE]",
            file=sys.stderr,
        )
        print(
            "  NOTES omitted → docs/release_notes/<VERSION>.md\n"
            "  NOTES = @path → read that file\n"
            "  NOTES = plain text → use as-is (legacy)",
            file=sys.stderr,
        )
        return 1

    version = sys.argv[1].lstrip("v")
    notes_arg = sys.argv[5] if len(sys.argv) > 5 else ""

    if not notes_arg:
        try:
            release_notes = load_release_notes(version).rstrip()
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    elif notes_arg.startswith("@"):
        path = Path(notes_arg[1:])
        if not path.is_file():
            print(f"Notes file not found: {path}", file=sys.stderr)
            return 1
        release_notes = path.read_text(encoding="utf-8").strip()
    else:
        release_notes = notes_arg

    manifest = {
        "version": version,
        "release_notes": release_notes,
        "downloads": {
            "windows": sys.argv[2],
            "macos_intel": sys.argv[3],
            "macos_arm64": sys.argv[4],
        },
    }
    json.dump(manifest, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
