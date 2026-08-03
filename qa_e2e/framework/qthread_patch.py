"""
Remap production QThread workers onto Python threads for Mac Intel CI.

Customer builds use QThread (IndexingWorker / SearchWorker / TileCropWorker).
On some macos-15-intel CI hosts, PyTorch/OpenMP inside a Qt QThread stalls
forever and holds the inference lock, so a later Python-thread fallback also
deadlocks.

This patch keeps the same ``run()`` bodies (real DINOv2 / FAISS / SQLite) but
starts them with ``threading.Thread`` instead of ``QThread.start``.
Production code is untouched.
"""

from __future__ import annotations

import logging
import threading
from typing import Type

from PySide6.QtCore import QThread

logger = logging.getLogger("tilevision.qa_e2e.qthread_patch")

_PATCHED = False


def install_python_thread_workers() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    from src.presentation.workers.indexing_worker import IndexingWorker
    from src.presentation.workers.search_worker import SearchWorker
    from src.presentation.workers.tile_crop_worker import TileCropWorker

    for cls in (IndexingWorker, SearchWorker, TileCropWorker):
        _patch_worker_class(cls)
    logger.info(
        "[QA] Remapped IndexingWorker/SearchWorker/TileCropWorker "
        "onto Python threads (real run() bodies)"
    )


def _patch_worker_class(cls: Type[QThread]) -> None:
    if getattr(cls, "_qa_python_thread_patched", False):
        return

    def start(self, *args, **kwargs) -> None:  # noqa: ANN001
        if getattr(self, "_qa_py_thread", None) and self._qa_py_thread.is_alive():
            return

        self._qa_finished = False

        # ViewModels connect finished → deleteLater. Emitting QThread.finished
        # from a non-QThread runner + deleteLater races and can SIGSEGV
        # ("shared QObject was deleted directly"). Production signals used by
        # the UI (search_completed / indexing_finished / …) still fire from
        # run(); we intentionally skip QThread.finished and no-op deleteLater.
        try:
            self.finished.disconnect()
        except Exception:
            pass

        def _runner() -> None:
            try:
                self.run()
            finally:
                self._qa_finished = True

        thread = threading.Thread(
            target=_runner,
            name=f"qa-{cls.__name__}",
            daemon=True,
        )
        self._qa_py_thread = thread
        thread.start()

    def isRunning(self) -> bool:  # noqa: ANN001
        t = getattr(self, "_qa_py_thread", None)
        return bool(t is not None and t.is_alive())

    def wait(self, msecs: int = 30000) -> bool:  # noqa: ANN001
        t = getattr(self, "_qa_py_thread", None)
        if t is None:
            return True
        t.join(timeout=max(0.0, msecs / 1000.0))
        return not t.is_alive()

    def deleteLater(self) -> None:  # noqa: ANN001
        # No-op during QA — Python GC owns lifetime of patched workers.
        return

    cls.start = start  # type: ignore[method-assign]
    cls.isRunning = isRunning  # type: ignore[method-assign]
    cls.wait = wait  # type: ignore[method-assign]
    cls.deleteLater = deleteLater  # type: ignore[method-assign]
    cls._qa_python_thread_patched = True
