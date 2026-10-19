"""Release notes file is the single source for update_manifest + GH release."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_load_release_notes_1_2_43():
    from scripts.load_release_notes import load_release_notes

    text = load_release_notes("1.2.43")
    assert "Faster, safer, clearer" in text
    assert "encrypted" in text.casefold()
    assert "Export Catalogue" in text
    assert "Help page" in text
    # H1 title stripped for in-app dialog
    assert not text.lstrip().startswith("#")


def test_generate_update_manifest_reads_notes_file(tmp_path):
    out = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_update_manifest.py"),
            "1.2.43",
            "https://example.com/win.exe",
            "https://example.com/intel.dmg",
            "https://example.com/arm.dmg",
        ],
        cwd=str(ROOT),
        text=True,
    )
    data = json.loads(out)
    assert data["version"] == "1.2.43"
    assert "encrypted" in data["release_notes"].casefold()
    assert "separate Intel and Apple Silicon" not in data["release_notes"]
    assert data["downloads"]["windows"].endswith("win.exe")


def test_generate_update_manifest_fails_without_notes(tmp_path, monkeypatch):
    # Point loader at empty notes dir via cwd copy is heavy — call loader directly.
    from scripts.load_release_notes import load_release_notes

    try:
        load_release_notes("9.9.9", root=tmp_path)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "9.9.9.md" in str(exc)
