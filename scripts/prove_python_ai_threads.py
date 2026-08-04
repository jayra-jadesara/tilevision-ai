"""CI prove step: production Mac AI workers use Python threads (OpenMP-safe)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow ``python scripts/prove_python_ai_threads.py`` without PYTHONPATH.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    from src.presentation.workers.native_ai_thread import (
        configure_macos_openmp_for_ai,
        install_python_ai_worker_threads,
        production_uses_python_ai_threads,
        should_use_python_ai_threads,
    )

    configure_macos_openmp_for_ai()
    install_python_ai_worker_threads()
    assert should_use_python_ai_threads(), "Darwin must prefer Python AI threads"
    assert production_uses_python_ai_threads(), "production install failed"
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    print("OK: production OpenMP-safe AI threads armed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Trigger Mac Intel Search Gate for production validation of v1.2.20.
