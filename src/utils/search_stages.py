"""
Search pipeline stage logging for TileVision AI reliability.

Every stage of drop → results should leave an INFO breadcrumb so field
support can see exactly where a search stopped. Stages never change
behavior — they only log (and optionally notify a progress callback).
"""

from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from typing import Callable, Optional

StageCallback = Optional[Callable[[str], None]]

# Canonical stage names (stable for support / diagnostics).
STAGE_DROP_ACCEPTED = "Drop accepted"
STAGE_DROP_REJECTED = "Drop rejected"
STAGE_HEALTH_OK = "Index health OK"
STAGE_IMAGE_DECODED = "Image decoded"
STAGE_PREPROCESS_COMPLETE = "Preprocess complete"
STAGE_EMBEDDING_GENERATED = "Embedding generated"
STAGE_EMBEDDING_NORMALIZED = "Embedding normalized"
STAGE_EMBEDDING_CACHE_HIT = "Embedding cache hit"
STAGE_FAISS_SEARCH = "FAISS search complete"
STAGE_SQLITE_HYDRATE = "SQLite metadata loaded"
STAGE_RERANK_COMPLETE = "Hybrid rerank complete"
STAGE_ORB_VERIFICATION = "ORB verification applied"
STAGE_WEAK_FILTER = "Weak-result filter applied"
STAGE_THUMBNAILS_QUEUED = "Thumbnails queued"
STAGE_RESULTS_READY = "Results ready for UI"
STAGE_FAILED = "Search stage failed"

# Warn + dump stack when a single stage wall time exceeds this.
_SLOW_STAGE_S = float(os.environ.get("TILEVISION_SEARCH_SLOW_STAGE_S", "1.0"))

_last_stage_mono: dict[int, float] = {}
_last_stage_name: dict[int, str] = {}


def _rss_mb() -> float | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: kB; macOS: bytes
        if usage > 10_000_000:
            return usage / (1024.0 * 1024.0)
        return usage / 1024.0
    except Exception:
        return None


def _cpu_percent_hint() -> str:
    """Best-effort one-shot CPU sample (may be 0.0 on first call)."""
    try:
        import psutil

        return f"{psutil.Process(os.getpid()).cpu_percent(interval=None):.1f}%"
    except Exception:
        return "n/a"


def log_search_stage(
    logger: logging.Logger,
    stage: str,
    *,
    detail: str = "",
    on_stage: StageCallback = None,
) -> None:
    """Log a search stage and optionally forward to the UI progress callback."""
    tid = threading.get_ident()
    now = time.monotonic()
    prev = _last_stage_mono.get(tid)
    prev_name = _last_stage_name.get(tid, "")
    delta_s = (now - prev) if prev is not None else None
    _last_stage_mono[tid] = now
    _last_stage_name[tid] = stage

    rss = _rss_mb()
    rss_txt = f"{rss:.1f}MiB" if rss is not None else "n/a"
    delta_txt = f"{delta_s:.3f}s" if delta_s is not None else "start"
    message = (
        f"[SEARCH] {stage} | tid={tid} Δ={delta_txt} "
        f"cpu={_cpu_percent_hint()} rss={rss_txt}"
    )
    if detail:
        message = f"{message} — {detail}"
    logger.info(message)

    if delta_s is not None and delta_s >= _SLOW_STAGE_S:
        logger.warning(
            "[SEARCH] SLOW STAGE preceding '%s': previous='%s' took %.3fs "
            "(threshold=%.1fs) tid=%s thread=%s\n%s",
            stage,
            prev_name,
            delta_s,
            _SLOW_STAGE_S,
            tid,
            threading.current_thread().name,
            "".join(traceback.format_stack(limit=20)),
        )

    if on_stage is not None:
        try:
            on_stage(stage if not detail else f"{stage}: {detail}")
        except Exception as exc:  # never break search for UI progress
            logger.debug("on_stage callback failed: %s", exc)


def log_search_failure(
    logger: logging.Logger,
    stage: str,
    error: BaseException | str,
) -> None:
    tid = threading.get_ident()
    logger.error(
        "[SEARCH] %s — %s: %s | tid=%s thread=%s",
        STAGE_FAILED,
        stage,
        error,
        tid,
        threading.current_thread().name,
    )
    if isinstance(error, BaseException):
        logger.error(
            "[SEARCH] exception traceback:\n%s",
            "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        )
