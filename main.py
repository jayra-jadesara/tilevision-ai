"""
TileVision AI — Application Entry Point.

This is the top-level script to launch the application.

Usage:
    python main.py
    TileVisionAI --verify-bundle
    TileVisionAI --release-validation [--profile pr|full] [--out DIR]

Design Decision:
    This script contains minimal logic — just enough to invoke the
    composition root in src/app.py and propagate the exit code.
    Keeping the entry point thin makes it easier to wrap with PyInstaller
    for distribution as a standalone Windows .exe / macOS .app.
"""

import sys
import os

# Ensure the project root is on sys.path so `src.*` imports resolve correctly
# when running directly with `python main.py` (not installed as a package).
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Mac Metal: enable CPU fallback for unimplemented MPS ops *before* torch is
# imported by the app. Without this, DINOv2 search crashes on upsample_bicubic2d.
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _run_release_validation() -> int:
    """
    Run the Release Validation Suite from the *packaged* binary.

    Used on CI after installing TileVision AI.app from the customer DMG:

        /Applications/TileVision AI.app/Contents/MacOS/TileVisionAI \\
            --release-validation --profile pr --out /tmp/rv_out

    The suite drivers live in the checkout (`TILEVISION_QA_SUITE_DIR`).
    Product modules (src, torch, models) come from the frozen .app — not
    from a source-tree `python` environment.
    """
    # Strip the mode flag so qa_e2e argparse sees only its own flags.
    sys.argv = [sys.argv[0], *sys.argv[2:]]

    suite_dir = os.environ.get("TILEVISION_QA_SUITE_DIR", "").strip()
    if suite_dir:
        # Parent that contains the `qa_e2e` package (usually the git checkout).
        if suite_dir not in sys.path:
            sys.path.append(suite_dir)
    elif not getattr(sys, "frozen", False):
        # Dev: qa_e2e lives next to main.py
        if _PROJECT_ROOT not in sys.path:
            sys.path.insert(0, _PROJECT_ROOT)

    # When frozen, keep MEIPASS / packaged `src` ahead of any checkout `src`.
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            while _PROJECT_ROOT in sys.path:
                sys.path.remove(_PROJECT_ROOT)
            if meipass in sys.path:
                sys.path.remove(meipass)
            sys.path.insert(0, meipass)
        os.environ.setdefault("TILEVISION_QA_PACKAGED_APP", "1")

    os.environ.setdefault("TILEVISION_DEV_MODE", "1")
    os.environ.setdefault("TILEVISION_OFFLINE_MODEL", "1")

    from qa_e2e.run_release_validation import main as rv_main

    return int(rv_main())


def main() -> None:
    """Application entry point. Delegates to the composition root."""
    if len(sys.argv) > 1 and sys.argv[1] == "--verify-bundle":
        import torch

        import torch.cuda  # noqa: F401 — required by torch.__init__ even on CPU-only PCs

        print(f"bundle OK torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "--release-validation":
        sys.exit(_run_release_validation())

    from src.ai.gpu_info import configure_mps_fallback

    configure_mps_fallback()

    from src.app import build_application
    exit_code = build_application()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
