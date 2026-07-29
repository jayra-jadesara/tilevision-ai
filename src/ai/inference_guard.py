"""
Thread-safety guard for shared AI inference resources.

Serializes DINOv2 forward passes and FAISS index mutations so background
indexing (QThread / folder monitor) cannot race with active search queries.

Search must NEVER wait forever behind indexing — use timed acquire.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger("tilevision.ai.inference_guard")

_INFERENCE_LOCK = threading.RLock()

# Search / query paths should fail fast if indexing holds the lock.
DEFAULT_SEARCH_LOCK_TIMEOUT_S = 25.0
# Indexing can wait longer for the lock (search should finish quickly).
DEFAULT_INDEX_LOCK_TIMEOUT_S = 120.0


class InferenceBusyError(TimeoutError):
    """Raised when the AI engine is busy (usually indexing) too long."""


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
            "Wait for Indexing to finish, then search again."
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
