"""
Thread-safety guard for shared AI inference resources.

Serializes DINOv2 forward passes and FAISS index mutations so background
indexing (QThread / folder monitor) cannot race with active search queries.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

_INFERENCE_LOCK = threading.RLock()


@contextmanager
def synchronized_inference() -> Iterator[None]:
    """Acquire the global inference lock for the duration of a block."""
    _INFERENCE_LOCK.acquire()
    try:
        yield
    finally:
        _INFERENCE_LOCK.release()
