"""
Search pipeline stage logging for TileVision AI reliability.

Every stage of drop → results should leave an INFO breadcrumb so field
support can see exactly where a search stopped. Stages never change
behavior — they only log (and optionally notify a progress callback).
"""

from __future__ import annotations

import logging
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
STAGE_WEAK_FILTER = "Weak-result filter applied"
STAGE_THUMBNAILS_QUEUED = "Thumbnails queued"
STAGE_RESULTS_READY = "Results ready for UI"
STAGE_FAILED = "Search stage failed"


def log_search_stage(
    logger: logging.Logger,
    stage: str,
    *,
    detail: str = "",
    on_stage: StageCallback = None,
) -> None:
    """Log a search stage and optionally forward to the UI progress callback."""
    message = f"[SEARCH] {stage}"
    if detail:
        message = f"{message} — {detail}"
    logger.info(message)
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
    logger.error("[SEARCH] %s — %s: %s", STAGE_FAILED, stage, error)
