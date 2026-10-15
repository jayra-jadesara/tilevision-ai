"""Indexing timeout helpers for rebuild / scan resilience."""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable, Optional, TypeVar

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
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_FILE_INDEX_TIMEOUT_S


class IndexFileTimeoutError(TimeoutError):
    """Raised when indexing exceeds the safety timeout (caller already returned)."""


class IndexStallAbort(RuntimeError):
    """
    Rebuild/scan must stop: a timed-out embed thread still holds the global
    inference RLock, so continuing would only time out again.
    """


def stall_abort_message(label: str, timeout_s: float) -> str:
    return (
        f"Indexing stalled on {label} after {timeout_s:.0f}s. "
        "A timed-out embed still holds the AI lock — restart the app, "
        "then rebuild."
    )


def run_with_timeout(fn: Callable[[], T], *, timeout_s: float, label: str) -> T:
    """
    Run ``fn`` on a daemon thread; raise IndexFileTimeoutError at ``timeout_s``.

    Important: do **not** use ``ThreadPoolExecutor`` as a context manager here —
    its ``shutdown(wait=True)`` would block until the hung task finishes, which
    defeats the timeout (measured: sleep(25) with timeout=5 still took 25s).

    A timed-out torch call cannot be force-killed and may keep holding
    ``_INFERENCE_LOCK``. Callers must abort the rebuild (see IndexStallAbort)
    rather than continue to the next file.
    """
    if timeout_s <= 0:
        return fn()

    box: dict = {}
    done = threading.Event()

    def _worker() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — re-raise on caller side
            box["exc"] = exc
        finally:
            done.set()

    thread = threading.Thread(
        target=_worker,
        name=f"index-timeout-{str(label)[:48]}",
        daemon=True,
    )
    thread.start()
    if not done.wait(timeout=float(timeout_s)):
        logger.error(
            "INDEX TIMEOUT: %s exceeded %.0fs — returning to caller "
            "(worker thread may still be running / holding the inference lock)",
            label,
            timeout_s,
        )
        raise IndexFileTimeoutError(
            f"Indexing timed out after {timeout_s:.0f}s: {label}"
        )

    if "exc" in box:
        raise box["exc"]
    return box.get("value")  # type: ignore[return-value]


def inference_lock_is_held() -> Optional[bool]:
    """Best-effort: True if the global inference lock looks held by someone."""
    try:
        from src.ai.inference_guard import inference_lock_held

        return bool(inference_lock_held())
    except Exception:
        return None
