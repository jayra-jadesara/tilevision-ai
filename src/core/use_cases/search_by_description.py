"""
Text / description search against the CLIP FAISS index.

Plain text-only in this first pass — no hybrid image+text queries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.ai.clip_embedder import ClipEmbedder
from src.ai.similarity_score import calibrate_display_percent
from src.ai.vector_index import FaissIndexManager
from src.core.models import SearchResult
from src.data.repository_interface import IImageRepository
from src.utils.image_utils import get_thumbnail_path
from src.utils.search_stages import log_search_stage

logger = logging.getLogger("tilevision.core.use_cases.search_by_description")


class SearchByDescriptionUseCase:
    """Embed a text description with CLIP and search the CLIP FAISS index."""

    def __init__(
        self,
        image_repository: IImageRepository,
        clip_embedder: ClipEmbedder,
        clip_index: FaissIndexManager,
        thumbnail_dir: str,
    ) -> None:
        self._repo = image_repository
        self._clip = clip_embedder
        self._index = clip_index
        self._thumbnail_dir = Path(thumbnail_dir)

    def execute(
        self,
        query_text: str,
        top_k: int = 20,
        filters: Optional[Dict[str, str]] = None,
        on_stage: Optional[Callable[[str], None]] = None,
    ) -> List[SearchResult]:
        text = (query_text or "").strip()
        if not text:
            raise ValueError("Enter a short description to search.")

        top_k = max(1, int(top_k))

        def stage(name: str, detail: str = "") -> None:
            log_search_stage(logger, name, detail=detail, on_stage=on_stage)

        stage("text_search_start", text[:80])

        # First-use CLIP download can take minutes on a normal connection —
        # report progress so the UI stall watchdog keeps seeing heartbeats
        # (via SearchViewModel._on_search_progress / search_heartbeat).
        needs_download = False
        try:
            from src.ai.clip_model_paths import clip_model_available_locally

            needs_download = (not self._clip.is_loaded) and (
                not clip_model_available_locally()
            )
        except Exception:
            needs_download = not self._clip.is_loaded
        if needs_download:
            stage(
                "Downloading description-search model",
                "one-time, ~580 MB",
            )

        def _clip_progress(message: str) -> None:
            if on_stage is not None:
                try:
                    on_stage(message)
                except Exception:
                    pass

        self._clip.ensure_loaded(on_progress=_clip_progress)
        if getattr(self._index, "_index", None) is None:
            self._index.load_index()

        total = self._index.get_total_count()
        if total <= 0:
            raise RuntimeError(
                "The description-search index is empty.\n\n"
                "Index (or rebuild) your catalogue so each tile also gets a "
                "CLIP embedding, then try again."
            )

        stage("text_embedding")
        query_vec = self._clip.embed_text(text)

        search_k = min(max(top_k * 5, top_k), total)
        stage("clip_faiss_search", f"k={search_k}")
        matching_ids, scores = self._index.search_vectors(query_vec, search_k)
        if not matching_ids:
            return []

        # Collapse duplicate ids (best score) — CLIP index stores one vector per tile.
        best: dict[int, float] = {}
        order: list[int] = []
        for tid, score in zip(matching_ids, scores):
            tid_i = int(tid)
            sc = float(score)
            if tid_i not in best:
                order.append(tid_i)
                best[tid_i] = sc
            elif sc > best[tid_i]:
                best[tid_i] = sc
        order.sort(key=lambda i: best[i], reverse=True)

        tiles = self._repo.get_by_ids(order[: max(top_k * 3, top_k)])
        tile_map = {t.id: t for t in tiles if t.id is not None}

        results: List[SearchResult] = []
        for tid in order:
            tile = tile_map.get(tid)
            if tile is None:
                continue
            if filters:
                if any(
                    getattr(tile, key, None) != value
                    for key, value in filters.items()
                    if value
                ):
                    continue
            thumb = get_thumbnail_path(Path(tile.file_path), self._thumbnail_dir)
            # CLIP IP scores are already cosine-like after L2 norm.
            display = calibrate_display_percent(
                max(0.0, min(1.0, best[tid])),
                exact_match=False,
            )
            results.append(
                SearchResult(
                    tile=tile,
                    similarity_score=display,
                    thumbnail_path=str(thumb),
                )
            )
            if len(results) >= top_k:
                break

        stage("text_results_ready", f"{len(results)}")
        return results
