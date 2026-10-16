"""
Text / description search against the CLIP FAISS index.

Plain text-only in this first pass — no hybrid image+text queries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.ai.clip_embedder import ClipEmbedder
from src.ai.search_quality.result_dedupe import collapse_scored_by_content
from src.ai.similarity_score import calibrate_clip_display_percent
from src.ai.vector_index import FaissIndexManager
from src.core.models import SearchResult, TileImage
from src.core.use_cases.clip_text_query import (
    apply_relevance_cutoff,
    average_l2_normalized,
    description_prompt_variants,
    score_distribution_stats,
)
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
        # Last-run meta for UI ("Showing N of M…") / low-confidence banner.
        self.last_result_meta: Dict[str, object] = {}

    def embed_text_ensemble(self, query_text: str) -> np.ndarray:
        """Average L2-normalized embeddings across prompt templates."""
        variants = description_prompt_variants(query_text)
        if not variants:
            raise ValueError("Enter a short description to search.")
        # One batched get_text_features for all templates (not N sequential).
        embed_texts = getattr(self._clip, "embed_texts", None)
        if callable(embed_texts):
            vectors = list(embed_texts(variants))
        else:
            vectors = [self._clip.embed_text(v) for v in variants]
        return average_l2_normalized(vectors)

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
        self.last_result_meta = {"query": text}

        def stage(name: str, detail: str = "") -> None:
            log_search_stage(logger, name, detail=detail, on_stage=on_stage)

        stage("text_search_start", text[:80])

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

        total_files = self._index.get_total_count()
        if total_files <= 0:
            raise RuntimeError(
                "The description-search index is empty.\n\n"
                "Index (or rebuild) your catalogue so each tile also gets a "
                "CLIP embedding, then try again."
            )

        stage("text_embedding", "prompt_ensemble")
        query_vec = self.embed_text_ensemble(text)

        # Oversample so cutoff still has room after filters + content dedupe.
        search_k = min(max(top_k * 8, top_k), total_files)
        stage("clip_faiss_search", f"k={search_k}")
        matching_ids, scores = self._index.search_vectors(query_vec, search_k)
        if not matching_ids:
            self.last_result_meta.update(
                {
                    "shown": 0,
                    "matched": 0,
                    "catalogue": total_files,
                    "catalogue_unique": 0,
                    "low_confidence": False,
                }
            )
            return []

        best: dict[int, float] = {}
        for tid, score in zip(matching_ids, scores):
            tid_i = int(tid)
            sc = float(score)
            prev = best.get(tid_i)
            if prev is None or sc > prev:
                best[tid_i] = sc

        # Hydrate tiles early so we can dedupe by sha256 *before* cutoff stats.
        ids = list(best.keys())
        tiles = self._repo.get_by_ids(ids)
        tile_map = {t.id: t for t in tiles if t.id is not None}
        scored_tiles: List[Tuple[TileImage, float]] = []
        for tid, sc in best.items():
            tile = tile_map.get(tid)
            if tile is None:
                continue
            scored_tiles.append((tile, sc))

        collapsed = collapse_scored_by_content(scored_tiles)
        unique_catalogue = len(collapsed)
        # Id-score pairs for cutoff (primary tile ids only).
        scored_pairs: List[Tuple[int, float]] = [
            (int(tile.id), sc)
            for tile, sc, _paths in collapsed
            if tile.id is not None
        ]
        path_by_id = {
            int(tile.id): paths
            for tile, _sc, paths in collapsed
            if tile.id is not None
        }

        raw_scores = [sc for _, sc in scored_pairs]
        stats = score_distribution_stats(raw_scores)
        n_templates = len(description_prompt_variants(text))
        logger.info(
            "description_search_scores query=%r templates=%d unique=%d files=%d "
            "top10=%s min=%s max=%s median=%s spread=%s above_floor=%s "
            "within_margin=%s margin=%s",
            text,
            n_templates,
            unique_catalogue,
            total_files,
            stats["top10"],
            stats["min"],
            stats["max"],
            stats["median"],
            stats["spread"],
            stats["above_floor"],
            stats["within_margin"],
            stats["margin"],
        )
        kept, used_fallback, before_cutoff = apply_relevance_cutoff(
            scored_pairs, top_k=max(top_k * 3, top_k)
        )
        order = [tid for tid, _ in kept]

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
            display = calibrate_clip_display_percent(
                max(0.0, min(1.0, best[tid])),
            )
            paths = path_by_id.get(tid) or [tile.file_path]
            results.append(
                SearchResult(
                    tile=tile,
                    similarity_score=display,
                    thumbnail_path=str(thumb),
                    copy_paths=list(paths),
                    extra_copy_count=max(0, len(paths) - 1),
                )
            )
            if len(results) >= top_k:
                break

        # If filters emptied a fallback set, re-fill from unfiltered cutoff.
        if not results and used_fallback:
            for tid, _sc in kept[:3]:
                tile = tile_map.get(tid) or next(
                    (t for t in self._repo.get_by_ids([tid]) if t.id == tid),
                    None,
                )
                if tile is None:
                    continue
                thumb = get_thumbnail_path(Path(tile.file_path), self._thumbnail_dir)
                display = calibrate_clip_display_percent(
                    max(0.0, min(1.0, best[tid])),
                )
                paths = path_by_id.get(tid) or [tile.file_path]
                results.append(
                    SearchResult(
                        tile=tile,
                        similarity_score=display,
                        thumbnail_path=str(thumb),
                        copy_paths=list(paths),
                        extra_copy_count=max(0, len(paths) - 1),
                    )
                )

        shown = len(results)
        status = self._status_text(
            text=text,
            shown=shown,
            unique_matched=before_cutoff,
            unique_catalogue=unique_catalogue,
            total_files=total_files,
        )
        self.last_result_meta.update(
            {
                "shown": shown,
                "matched": before_cutoff,
                "catalogue": total_files,
                "catalogue_unique": unique_catalogue,
                "low_confidence": used_fallback,
                "status": status,
            }
        )
        stage(
            "text_results_ready",
            f"{shown} (matched_unique={before_cutoff} unique={unique_catalogue} "
            f"files={total_files} query={text!r} low_confidence={used_fallback})",
        )
        return results

    @staticmethod
    def _status_text(
        *,
        text: str,
        shown: int,
        unique_matched: int,
        unique_catalogue: int,
        total_files: int,
    ) -> str:
        """Unique-tile counts with optional file-copy parenthetical."""
        base_n = unique_matched if unique_matched > 0 else unique_catalogue
        if total_files > unique_catalogue > 0:
            suffix = f" ({total_files} files incl. copies)"
        else:
            suffix = ""
        if shown < base_n:
            return (
                f"Showing {shown} of {base_n} tiles that match '{text}'{suffix}"
            )
        return f"Showing {shown} tiles that match '{text}'{suffix}"
