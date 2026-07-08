"""
TileVision AI — Application Entry Point.

This is the top-level script to launch the application.

Usage:
    python main.py

Design Decision:
    This script contains minimal logic — just enough to invoke the
    composition root in src/app.py and propagate the exit code.
    Keeping the entry point thin makes it easier to wrap with PyInstaller
    for distribution as a standalone Windows .exe.
"""

import sys
import os

# Ensure the project root is on sys.path so `src.*` imports resolve correctly
# when running directly with `python main.py` (not installed as a package).
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main() -> None:
    """Application entry point. Delegates to the composition root."""
    from src.app import build_application
    exit_code = build_application()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
