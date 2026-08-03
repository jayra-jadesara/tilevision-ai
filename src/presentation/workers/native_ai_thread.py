"""
Safe AI worker threads for TileVision AI.

ROOT CAUSE (macOS Intel search hang)
------------------------------------
DINOv2 / FAISS use OpenMP (libomp / Accelerate). When those OpenMP parallel
regions run inside a Qt ``QThread`` on macOS Intel, the OpenMP runtime can
stall forever and never return from ``torch`` / ``faiss`` calls. The UI then
eventually shows a timeout/stall dialog even though the hang is not a slow
search — the worker thread never completes.

Windows and Apple Silicon do not exhibit this OpenMP↔QThread deadlock with
our current torch builds. Intel Mac is the affected platform.

FIX
---
On Darwin, run ``IndexingWorker`` / ``SearchWorker`` / ``TileCropWorker``
``run()`` bodies on a Python ``threading.Thread`` while keeping the same
``QObject`` signals (QueuedConnection to the UI). Pipeline, AI models, FAISS,
SQLite, and hybrid rerank are unchanged.

This is the same approach proven in QA CI (`qa_e2e/framework/qthread_patch.py`)
— promoted to production so customers get the fix, not only CI.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Type

from PySide6.QtCore import QThread

logger = logging.getLogger("tilevision.presentation.workers.native_ai_thread")

_INSTALLED = False
_ENV_FORCE_PYTHON = "TILEVISION_FORCE_PYTHON_AI_THREADS"
_ENV_FORCE_QTHREAD = "TILEVISION_FORCE_QTHREAD_AI"


def should_use_python_ai_threads() -> bool:
    """
    True when AI workers must avoid Qt QThread for OpenMP safety.

    Override:
      TILEVISION_FORCE_PYTHON_AI_THREADS=1  → always Python threads
      TILEVISION_FORCE_QTHREAD_AI=1         → always Qt QThread (debug only)
    """
    if os.environ.get(_ENV_FORCE_QTHREAD, "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if os.environ.get(_ENV_FORCE_PYTHON, "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    # All Darwin: Intel is the proven hang; Silicon query also uses CPU torch
    # and shares the same OpenMP risk class under QThread.
    return sys.platform == "darwin"


def configure_macos_openmp_for_ai() -> None:
    """
    Cap OpenMP / BLAS threads on macOS Intel before libomp initializes.

    Must run as early as possible (before first torch/faiss import is ideal).
    Does not reduce Apple Silicon thread budgets unless env already set.
    """
    if sys.platform != "darwin":
        return

    from src.utils.platform_info import is_mac_intel

    # Safe defaults on every Mac — never overwrite an explicit user/CI setting.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

    if not is_mac_intel():
        return

    # Intel Mac: oversubscription (torch 8 × faiss 8 × Qt thread) deadlocks.
    # Force — do not trust prior env that may have set a high thread count.
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    logger.info(
        "macOS Intel OpenMP capped to 1 thread (avoids QThread/OpenMP deadlock)"
    )


def apply_torch_faiss_thread_caps() -> None:
    """Apply torch/faiss thread caps after those packages are importable."""
    if sys.platform != "darwin":
        return
    from src.utils.platform_info import is_mac_intel

    if not is_mac_intel():
        return
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    except Exception as exc:
        logger.debug("torch thread cap skipped: %s", exc)
    try:
        import faiss

        faiss.omp_set_num_threads(1)
    except Exception as exc:
        logger.debug("faiss thread cap skipped: %s", exc)


def install_python_ai_worker_threads() -> None:
    """
    Patch AI QThread subclasses so ``start()`` uses a Python thread on Darwin.

    Idempotent. Safe to call from app startup and from QA harness.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    if not should_use_python_ai_threads():
        logger.info("AI workers keep Qt QThread (platform=%s)", sys.platform)
        _INSTALLED = True
        return

    from src.presentation.workers.indexing_worker import IndexingWorker
    from src.presentation.workers.search_worker import SearchWorker

    # Only workers that run PyTorch/FAISS OpenMP. TileCropWorker stays on
    # QThread (OpenCV/ONNX) so Qt signal affinity in unit tests remains intact.
    for cls in (IndexingWorker, SearchWorker):
        _patch_worker_class(cls)

    _INSTALLED = True
    logger.info(
        "Search/Indexing AI workers use Python threads on Darwin "
        "(OpenMP-safe; same run() bodies / signals)"
    )


def production_uses_python_ai_threads() -> bool:
    """Whether production install has remapped AI workers (for QA skip)."""
    return _INSTALLED and should_use_python_ai_threads()


def _patch_worker_class(cls: Type[QThread]) -> None:
    if getattr(cls, "_tv_python_ai_thread_patched", False):
        return

    original_start = cls.start

    def start(self, *args, **kwargs) -> None:  # noqa: ANN001
        if not should_use_python_ai_threads():
            return original_start(self, *args, **kwargs)

        existing = getattr(self, "_tv_py_thread", None)
        if existing is not None and existing.is_alive():
            return

        self._tv_finished = False

        # ViewModels connect finished → deleteLater. Emitting QThread.finished
        # from a non-QThread runner + deleteLater can SIGSEGV. Production
        # completion signals (search_completed / indexing_finished / …) still
        # fire from run(); skip QThread.finished and no-op deleteLater.
        try:
            self.finished.disconnect()
        except Exception:
            pass

        def _runner() -> None:
            try:
                logger.info(
                    "%s Python AI thread started (tid=%s)",
                    cls.__name__,
                    threading.get_ident(),
                )
                self.run()
            finally:
                self._tv_finished = True
                logger.info(
                    "%s Python AI thread finished (tid=%s)",
                    cls.__name__,
                    threading.get_ident(),
                )

        thread = threading.Thread(
            target=_runner,
            name=f"tv-{cls.__name__}",
            daemon=True,
        )
        self._tv_py_thread = thread
        thread.start()

    def isRunning(self) -> bool:  # noqa: ANN001
        if not should_use_python_ai_threads():
            return QThread.isRunning(self)
        t = getattr(self, "_tv_py_thread", None)
        return bool(t is not None and t.is_alive())

    def isFinished(self) -> bool:  # noqa: ANN001
        if not should_use_python_ai_threads():
            return QThread.isFinished(self)
        if getattr(self, "_tv_py_thread", None) is None:
            return False
        return bool(getattr(self, "_tv_finished", False))

    def wait(self, msecs: int = 30000) -> bool:  # noqa: ANN001
        if not should_use_python_ai_threads():
            return QThread.wait(self, msecs)
        t = getattr(self, "_tv_py_thread", None)
        if t is None:
            return True
        t.join(timeout=max(0.0, msecs / 1000.0))
        return not t.is_alive()

    def deleteLater(self) -> None:  # noqa: ANN001
        if should_use_python_ai_threads():
            # Lifetime owned by ViewModel refs + Python GC.
            return
        return QThread.deleteLater(self)

    cls.start = start  # type: ignore[method-assign]
    cls.isRunning = isRunning  # type: ignore[method-assign]
    cls.isFinished = isFinished  # type: ignore[method-assign]
    cls.wait = wait  # type: ignore[method-assign]
    cls.deleteLater = deleteLater  # type: ignore[method-assign]
    cls._tv_python_ai_thread_patched = True
