"""
Remap production QThread workers onto Python threads for Mac Intel CI.

Customer builds now install the same OpenMP-safe Python-thread path in
``src.presentation.workers.native_ai_thread`` on Darwin. This QA helper
becomes a no-op when production already remapped the workers; otherwise it
applies the same patch so older checkouts still run on macos-15-intel CI.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("tilevision.qa_e2e.qthread_patch")

_PATCHED = False


def install_python_thread_workers() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    from src.presentation.workers.native_ai_thread import (
        install_python_ai_worker_threads,
        production_uses_python_ai_threads,
        should_use_python_ai_threads,
    )

    install_python_ai_worker_threads()
    if production_uses_python_ai_threads() or should_use_python_ai_threads():
        logger.info(
            "[QA] AI workers use production OpenMP-safe Python threads "
            "(IndexingWorker / SearchWorker / TileCropWorker)"
        )
        return

    logger.info("[QA] Platform keeps Qt QThread for AI workers")
