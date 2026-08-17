"""
Thread-safety guard for shared AI inference resources.

Serializes DINOv2 forward passes and FAISS index mutations so background
indexing (QThread / folder monitor) cannot race with active search queries.

Search must NEVER wait forever behind indexing — use timed acquire.
When the user drops an image to search, indexing must yield so results can return.

Background query warmup must NOT take this lock: a 40–50s dummy DINOv2
forward held it on Windows and blocked ``get_total_count()`` (Search
priority ON → Starting worker = 44s) even for catalog-cache hits that
never need DINOv2.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger("tilevision.ai.inference_guard")

_INFERENCE_LOCK = threading.RLock()

# Search / query paths wait for the lock; indexing must yield quickly.
DEFAULT_SEARCH_LOCK_TIMEOUT_S = 120.0
# Indexing can wait longer for the lock (search should finish quickly).
DEFAULT_INDEX_LOCK_TIMEOUT_S = 180.0

# Cooperative yield: indexing waits between work units while search is active.
_search_priority_count = 0
_search_priority_lock = threading.Lock()
_search_idle = threading.Event()
_search_idle.set()

# Background warmup: thread-local so only that thread skips the inference lock.
_warmup_tls = threading.local()
_warmup_in_progress = threading.Event()


class InferenceBusyError(TimeoutError):
    """Raised when the AI engine is busy (usually indexing) too long."""


def interactive_cpu_thread_count() -> int:
    """Intra-op threads for a user-facing search (not background warmup)."""
    try:
        from src.utils.platform_info import is_mac_intel

        if is_mac_intel():
            return 1
    except Exception:
        pass
    return min(8, os.cpu_count() or 4)


def _torch_thread_count() -> int | None:
    try:
        import torch

        return int(torch.get_num_threads())
    except Exception:
        return None


def restore_interactive_torch_threads() -> None:
    """Give a real search the full CPU budget even if warmup reduced it."""
    try:
        import torch

        target = interactive_cpu_thread_count()
        current = int(torch.get_num_threads())
        if current != target:
            torch.set_num_threads(target)
            logger.info(
                "Search restored torch intra-op threads %s → %s",
                current,
                target,
            )
    except Exception as exc:
        logger.debug("Could not restore interactive torch threads: %s", exc)


def is_warmup_compute() -> bool:
    """True only on the background warmup thread inside warmup_compute_scope."""
    return bool(getattr(_warmup_tls, "active", False))


def warmup_in_progress() -> bool:
    """True while any warmup_compute_scope is entered."""
    return _warmup_in_progress.is_set()


def _lower_os_thread_priority() -> int | None:
    """Lower this OS thread's priority. Returns previous Windows priority or None."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = k32.GetCurrentThread()
        previous = int(k32.GetThreadPriority(handle))
        thread_priority_below_normal = -1
        if k32.SetThreadPriority(handle, thread_priority_below_normal):
            logger.info("Warmup OS thread priority BELOW_NORMAL (was %s)", previous)
            return previous
        logger.debug("SetThreadPriority failed err=%s", k32.GetLastError())
    except Exception as exc:
        logger.debug("Warmup OS thread priority not set: %s", exc)
    return None


def _restore_os_thread_priority(previous: int | None) -> None:
    if previous is None or sys.platform != "win32":
        return
    try:
        import ctypes

        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        k32.SetThreadPriority(k32.GetCurrentThread(), int(previous))
    except Exception:
        pass


@contextmanager
def warmup_compute_scope(*, torch_threads: int = 1) -> Iterator[None]:
    """
    Limit CPU used by background query warmup and skip the inference lock.

    ``torch.set_num_threads`` is process-global: warmup uses 1 intra-op
    thread so a concurrent catalog search is not starved on a 4-core CPU.
    Search priority does not restore the full budget until this scope
    exits — otherwise the in-flight dummy forward would take every core
    again.
    """
    _warmup_tls.active = True
    _warmup_in_progress.set()
    prev_os = _lower_os_thread_priority()
    prev_torch: int | None = None
    try:
        import torch

        prev_torch = int(torch.get_num_threads())
        target = max(1, int(torch_threads))
        if prev_torch != target:
            torch.set_num_threads(target)
        logger.info(
            "Warmup compute scope ON (torch_threads %s → %s, skip inference lock)",
            prev_torch,
            torch.get_num_threads(),
        )
    except Exception as exc:
        logger.debug("Warmup torch thread cap skipped: %s", exc)
    try:
        yield
    finally:
        _restore_os_thread_priority(prev_os)
        _warmup_tls.active = False
        _warmup_in_progress.clear()
        if search_priority_active():
            restore_interactive_torch_threads()
        elif prev_torch is not None:
            try:
                import torch

                torch.set_num_threads(prev_torch)
            except Exception:
                pass
        logger.info("Warmup compute scope OFF")


def begin_search_priority() -> None:
    """Mark that a user search is starting — indexing / warmup should yield."""
    global _search_priority_count
    with _search_priority_lock:
        _search_priority_count += 1
        _search_idle.clear()
    # torch.set_num_threads is process-global. Restoring the full budget
    # while warmup is mid-forward would give that dummy oneDNN pass every
    # core again and starve a catalog-cache-hit search. Leave the cap in
    # place until warmup_compute_scope exits, then restore.
    if not warmup_in_progress():
        restore_interactive_torch_threads()
    logger.info(
        "Search priority ON (active=%s inference_lock_held=%s "
        "warmup_in_progress=%s torch_threads=%s cpu_count=%s "
        "active_threads=%s)",
        _search_priority_count,
        inference_lock_held(),
        warmup_in_progress(),
        _torch_thread_count(),
        os.cpu_count(),
        threading.active_count(),
    )


def end_search_priority() -> None:
    """Clear search priority when the search UI finishes (success/fail/timeout)."""
    global _search_priority_count
    with _search_priority_lock:
        _search_priority_count = max(0, _search_priority_count - 1)
        if _search_priority_count == 0:
            _search_idle.set()
    logger.info("Search priority OFF (active=%s)", _search_priority_count)


def search_priority_active() -> bool:
    """True while at least one search has requested priority."""
    return not _search_idle.is_set()


def wait_while_search_priority(*, max_wait_s: float = 180.0) -> None:
    """
    Indexing calls this between images/batches so drop-search can run.

    Blocks until search priority clears or max_wait_s elapses.
    """
    if _search_idle.is_set():
        return
    logger.info(
        "Indexing yielding to Search (wait up to %.0fs)...",
        max_wait_s,
    )
    cleared = _search_idle.wait(timeout=float(max_wait_s))
    if not cleared:
        logger.warning(
            "Indexing resume: search priority still set after %.0fs",
            max_wait_s,
        )


@contextmanager
def synchronized_inference(
    *,
    timeout: float | None = None,
    purpose: str = "inference",
) -> Iterator[None]:
    """
    Acquire the global inference lock for the duration of a block.

    Args:
        timeout: Seconds to wait for the lock. ``None`` waits forever
            (avoid for search). ``0`` tries once without blocking.
        purpose: Label for logs / error messages.
    """
    t0 = time.monotonic()
    if timeout is None:
        acquired = _INFERENCE_LOCK.acquire(blocking=True)
    else:
        acquired = _INFERENCE_LOCK.acquire(blocking=True, timeout=float(timeout))
    waited = time.monotonic() - t0
    if waited >= 0.1:
        logger.info(
            "Inference lock wait %.3fs purpose=%s warmup_in_progress=%s "
            "search_priority=%s",
            waited,
            purpose,
            warmup_in_progress(),
            search_priority_active(),
        )

    if not acquired:
        message = (
            f"AI engine busy ({purpose}) — could not start within {timeout:.0f}s. "
            "Wait for Indexing to finish (or pause it), then drop your image again."
        )
        logger.warning(message)
        raise InferenceBusyError(message)

    try:
        yield
    finally:
        _INFERENCE_LOCK.release()


def inference_lock_held() -> bool:
    """Best-effort: True if another thread likely holds the lock (not owned by us)."""
    # RLock has no public "is_locked by other"; try non-blocking acquire.
    acquired = _INFERENCE_LOCK.acquire(blocking=False)
    if acquired:
        _INFERENCE_LOCK.release()
        return False
    return True


def wait_until_inference_idle(*, max_wait_s: float = 90.0, poll_s: float = 0.15) -> bool:
    """
    Block until the global inference lock is free (or timeout).

    Used by Search before starting DINOv2 so drop-search is not stuck behind
    the current indexing forward pass.
    """
    deadline = time.monotonic() + float(max_wait_s)
    while time.monotonic() < deadline:
        if not inference_lock_held():
            return True
        time.sleep(float(poll_s))
    return not inference_lock_held()
