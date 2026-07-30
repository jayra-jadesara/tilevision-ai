"""
Thread-safety guard for shared AI inference resources.

Serializes DINOv2 forward passes and FAISS index mutations so background
indexing (QThread / folder monitor) cannot race with active search queries.

Search must NEVER wait forever behind indexing — use timed acquire.
When the user drops an image to search, indexing must yield so results can return.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger("tilevision.ai.inference_guard")

_INFERENCE_LOCK = threading.RLock()

# Search / query paths should fail fast if indexing holds the lock.
DEFAULT_SEARCH_LOCK_TIMEOUT_S = 60.0
# Indexing can wait longer for the lock (search should finish quickly).
DEFAULT_INDEX_LOCK_TIMEOUT_S = 120.0

# Cooperative yield: indexing waits between work units while search is active.
_search_priority_count = 0
_search_priority_lock = threading.Lock()
_search_idle = threading.Event()
_search_idle.set()


class InferenceBusyError(TimeoutError):
    """Raised when the AI engine is busy (usually indexing) too long."""


def begin_search_priority() -> None:
    """Mark that a user search is starting — indexing should yield."""
    global _search_priority_count
    with _search_priority_lock:
        _search_priority_count += 1
        _search_idle.clear()
    logger.info("Search priority ON (active=%s)", _search_priority_count)


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
    if timeout is None:
        acquired = _INFERENCE_LOCK.acquire(blocking=True)
    else:
        acquired = _INFERENCE_LOCK.acquire(blocking=True, timeout=float(timeout))

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
