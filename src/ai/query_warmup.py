"""
Background crop-tool query-path warmup.

Must not block MainWindow from appearing. A user who searches before this
finishes pays the same first-click cost as before PR #49 — but the app
itself is usable immediately.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger("tilevision.ai.query_warmup")

_ENV_SHAPES = "TILEVISION_WARMUP_SHAPES"


def warmup_shapes_from_env() -> tuple[int, ...]:
    """
    Which query batch shapes to prime.

    Default is n=1 only (Auto / Precise Crop). n=2 is a separate Windows
    oneDNN compile (~the same cost as n=1) and is deferred unless
    ``TILEVISION_WARMUP_SHAPES=1,2``.
    """
    raw = os.environ.get(_ENV_SHAPES, "1").strip()
    if not raw:
        return (1,)
    shapes: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part in {"1", "2"} and int(part) not in shapes:
            shapes.append(int(part))
    return tuple(shapes) or (1,)


def write_dummy_clean_tile(path: Path | None = None) -> Path:
    """Write a full-frame clean-tile JPEG matching Auto Crop n=1 input."""
    if path is None:
        handle = tempfile.NamedTemporaryFile(
            prefix="tv_warmup_autocrop_",
            suffix=".jpg",
            delete=False,
        )
        path = Path(handle.name)
        handle.close()
    rng = np.random.default_rng(7)
    edge = 600
    # Dense texture + square aspect → QueryKind.CLEAN_TILE / full_frame n=1.
    # Sparse line-on-white dummies classify as partial and wrongly take n=2.
    base = rng.integers(170, 220, size=(edge, edge, 3), dtype=np.uint8)
    base[::18, :] = np.clip(base[::18, :].astype(np.int16) - 35, 0, 255).astype(np.uint8)
    base[:, ::18] = np.clip(base[:, ::18].astype(np.int16) - 35, 0, 255).astype(np.uint8)
    Image.fromarray(base).save(path, quality=90)
    return path


def run_query_path_warmup(
    feature_extractor: Any,
    *,
    vector_index: Any | None = None,
    shapes: tuple[int, ...] | None = None,
) -> dict[str, float]:
    """
    Prime the crop-tool query path. Safe to call from a background thread.

    Aborts remaining steps if a user search has claimed priority.
    """
    from src.ai.inference_guard import search_priority_active, warmup_compute_scope

    if search_priority_active():
        logger.info("Query-path warm-up skipped — search already running")
        return {}

    shapes = shapes or warmup_shapes_from_env()
    timings: dict[str, float] = {}
    with warmup_compute_scope(torch_threads=1):
        if search_priority_active():
            logger.info("Query-path warm-up skipped — search already running")
            return {}
        warmup = getattr(feature_extractor, "warmup_query_inference", None)
        if callable(warmup):
            result = warmup(shapes=shapes)
            if isinstance(result, dict):
                timings.update(result)

        if (
            vector_index is not None
            and not search_priority_active()
            and hasattr(vector_index, "search_vectors")
            and hasattr(vector_index, "get_total_count")
        ):
            try:
                if int(vector_index.get_total_count()) > 0:
                    import time as _time

                    features = getattr(feature_extractor, "_last_query_features", None)
                    embedding = (
                        getattr(features, "embedding", None) if features is not None else None
                    )
                    if embedding is None:
                        dim = 1024
                        dim_fn = getattr(vector_index, "embedding_dimension", None)
                        if callable(dim_fn):
                            dim = int(dim_fn())
                        embedding = np.zeros(dim, dtype=np.float32)
                        embedding[0] = 1.0
                    t0 = _time.perf_counter()
                    vector_index.search_vectors(embedding, top_k=5)
                    timings["faiss_ms"] = (_time.perf_counter() - t0) * 1000.0
                    logger.info("Query-path warm-up faiss: %.0f ms", timings["faiss_ms"])
            except Exception as exc:
                logger.debug("Query-path FAISS warm-up skipped: %s", exc)

    logger.info(
        "Query-path warm-up finished (%s)",
        ", ".join(f"{k}={v:.0f}" for k, v in timings.items()) or "no-op",
    )
    return timings


def start_background_query_warmup(
    feature_extractor: Any,
    *,
    vector_index: Any | None = None,
) -> threading.Thread:
    """Start query-path warmup on a daemon thread. Returns immediately."""

    def _run() -> None:
        try:
            run_query_path_warmup(feature_extractor, vector_index=vector_index)
        except Exception as exc:
            logger.warning(
                "Background query-path warm-up failed (first search may pay cold start): %s",
                exc,
            )

    thread = threading.Thread(
        target=_run,
        name="tv-query-warmup",
        daemon=True,
    )
    thread.start()
    logger.info("Query-path warm-up started in background (UI is not blocked).")
    return thread
