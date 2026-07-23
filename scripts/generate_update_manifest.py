#!/usr/bin/env python3
"""Generate update_manifest.json for GitHub Releases."""

from __future__ import annotations

import json
import sys


def main() -> int:
    if len(sys.argv) < 5:
        print(
            "Usage: generate_update_manifest.py VERSION WINDOWS_URL "
            "MACOS_INTEL_URL MACOS_ARM64_URL [NOTES]",
            file=sys.stderr,
        )
        return 1

    version = sys.argv[1].lstrip("v")
    manifest = {
        "version": version,
        "release_notes": sys.argv[5] if len(sys.argv) > 5 else "",
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
