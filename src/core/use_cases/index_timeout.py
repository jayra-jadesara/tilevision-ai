"""Per-file indexing timeout helpers for rebuild / scan resilience."""

from __future__ import annotations

import concurrent.futures
import logging
import os
from typing import Callable, TypeVar

logger = logging.getLogger("tilevision.core.use_cases.index_timeout")

T = TypeVar("T")

# One bad PNG / hung forward must not block a whole rebuild. Override via env
# for tests (short) or slow machines (longer).
DEFAULT_FILE_INDEX_TIMEOUT_S = 90.0


def file_index_timeout_s() -> float:
    raw = os.environ.get("TILEVISION_INDEX_FILE_TIMEOUT_S", "").strip()
    if not raw:
        return DEFAULT_FILE_INDEX_TIMEOUT_S
    try:
        return max(5.0, float(raw))
    except ValueError:
        return DEFAULT_FILE_INDEX_TIMEOUT_S


class IndexFileTimeoutError(TimeoutError):
    """Raised when indexing one file exceeds the safety timeout."""


def run_with_timeout(fn: Callable[[], T], *, timeout_s: float, label: str) -> T:
    """
    Run ``fn`` in a worker thread; raise IndexFileTimeoutError on timeout.

    Note: a timed-out thread may keep running (torch cannot be force-killed).
    Callers should count the file as failed and continue; Cancel remains available.
    """
    if timeout_s <= 0:
        return fn()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="index-file"
    ) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=float(timeout_s))
        except concurrent.futures.TimeoutError as exc:
            logger.error(
                "INDEX FILE TIMEOUT: %s exceeded %.0fs — skipping/failing this file",
                label,
                timeout_s,
            )
            raise IndexFileTimeoutError(
                f"Indexing timed out after {timeout_s:.0f}s: {label}"
            ) from exc
