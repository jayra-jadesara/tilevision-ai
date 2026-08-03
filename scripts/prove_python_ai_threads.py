"""CI prove step: production Mac AI workers use Python threads (OpenMP-safe)."""

from __future__ import annotations

import os
import sys


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
