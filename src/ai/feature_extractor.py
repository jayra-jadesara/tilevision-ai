"""
Central AI feature extraction service.

This class is the ONLY place that knows how AI features are generated.

Pipeline

Image
   │
   ▼
ImagePreprocessor
   │
   ▼
DINOv2
   │
   ▼
HSV
   │
   ▼
LBP
   │
   ▼
Edge
   │
   ▼
Dominant Color
   │
   ▼
TileFeatures

Author:
TileVision AI v2
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, TYPE_CHECKING

import cv2
import numpy as np

from src.ai.embedder import DINOv2Embedder
from src.ai.models import TileFeatures, PreprocessedImage
from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.preprocess.index_primary import (
    finalize_index_pil,
    prepare_index_primary,
)
from src.ai.descriptors.color_descriptor import ColorDescriptor
from src.ai.descriptors.texture_descriptor import TextureDescriptor
from src.ai.descriptors.edge_descriptor import EdgeDescriptor
from src.ai.descriptors.pattern_descriptor import PatternDescriptor

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger("tilevision.ai.feature_extractor")


@dataclass(slots=True)
class ExtractTimings:
    preprocessing: float = 0.0
    dinov2: float = 0.0
    descriptors: float = 0.0
    total: float = 0.0


class FeatureExtractor:

    def __init__(
        self,
        embedder: DINOv2Embedder | None = None,
        *,
        preprocess_workers: int = 4,
    ):
        self._embedder = embedder or DINOv2Embedder()
        self._preprocess_workers = max(1, int(preprocess_workers))
        self._last_timings = ExtractTimings()
        self._last_query_features = None

    @property
    def last_timings(self) -> ExtractTimings:
        return self._last_timings

    # --------------------------------------------------------
    
    def load_model(self) -> None:
        self._embedder.load_model()

    def warmup_query_inference(
        self,
        *,
        shapes: tuple[int, ...] = (1,),
    ) -> dict[str, float]:
        """
        Prime the real Auto Crop query path (analyzer + letterbox + DINOv2 +
        descriptors). Logs n=1 / n=2 timings separately.

        Default ``shapes=(1,)`` — n=2 is a distinct Windows cold compile and
        must not run on the UI thread at launch.
        """
        from src.ai.inference_guard import search_priority_active
        from src.ai.query_warmup import write_dummy_clean_tile

        timings: dict[str, float] = {}
        if search_priority_active():
            logger.info("Query-path warm-up skipped — search already running")
            return timings

        dummy_path: Path | None = None
        if 1 in shapes:
            try:
                dummy_path = write_dummy_clean_tile()
                t0 = time.perf_counter()
                features, _embeddings = self.extract_for_search(
                    str(dummy_path),
                    query_origin="crop_tool",
                )
                timings["n1_ms"] = (time.perf_counter() - t0) * 1000.0
                timings["n1_dinov2_ms"] = self._last_timings.dinov2 * 1000.0
                timings["n1_preprocess_ms"] = self._last_timings.preprocessing * 1000.0
                timings["n1_descriptors_ms"] = self._last_timings.descriptors * 1000.0
                self._last_query_features = features
                logger.info(
                    "Query-path warm-up n=1: %.0f ms "
                    "(dinov2=%.2fs preprocess=%.2fs descriptors=%.2fs)",
                    timings["n1_ms"],
                    self._last_timings.dinov2,
                    self._last_timings.preprocessing,
                    self._last_timings.descriptors,
                )
            except Exception as exc:
                logger.warning(
                    "Query-path n=1 extract_for_search warm-up failed (%s) — "
                    "falling back to embedder batch",
                    exc,
                )
                warmup = getattr(self._embedder, "warmup_query_inference", None)
                if callable(warmup):
                    result = warmup(shapes=(1,))
                    if isinstance(result, dict):
                        timings.update(result)
                else:
                    dummy = DINOv2Embedder.dummy_query_view()
                    batch_fn = getattr(self._embedder, "extract_query_views_batch", None)
                    if callable(batch_fn):
                        t0 = time.perf_counter()
                        batch_fn([dummy])
                        timings["n1_ms"] = (time.perf_counter() - t0) * 1000.0
                        logger.info("Query-path warm-up n=1: %.0f ms", timings["n1_ms"])
            finally:
                if dummy_path is not None:
                    try:
                        dummy_path.unlink(missing_ok=True)
                    except Exception:
                        pass

        if 2 in shapes:
            if search_priority_active():
                logger.info("Query-path warm-up n=2: skipped — search requested")
            else:
                dummy = DINOv2Embedder.dummy_query_view()
                batch_fn = getattr(self._embedder, "extract_query_views_batch", None)
                if callable(batch_fn):
                    t0 = time.perf_counter()
                    batch_fn([dummy, dummy])
                    timings["n2_ms"] = (time.perf_counter() - t0) * 1000.0
                    logger.info("Query-path warm-up n=2: %.0f ms", timings["n2_ms"])
        else:
            logger.info(
                "Query-path warm-up n=2: skipped "
                "(first 2-view Manual Crop may pay a one-time cost)"
            )

        return timings

    @staticmethod
    def dominant_color(image_bgr):
        return ColorDescriptor.dominant_color_rgb(image_bgr)

    # --------------------------------------------------------

    def extract_descriptors_from_preprocessed(
        self,
        image: PreprocessedImage,
    ) -> tuple:
        """Return handcrafted descriptors from a preprocessed image."""
        color_hist = ColorDescriptor.extract(image.bgr)
        texture_hist = TextureDescriptor.extract(image.bgr)
        edge_hist = EdgeDescriptor.extract(image.bgr)
        pattern_features = PatternDescriptor.extract(image.bgr)
        dominant = self.dominant_color(image.bgr)
        return color_hist, texture_hist, edge_hist, pattern_features, dominant

    def extract_from_preprocessed(
        self,
        image: PreprocessedImage,
        *,
        for_query: bool = False,
    ) -> TileFeatures:
        """Extract full features when the image is already preprocessed."""
        total_start = time.perf_counter()

        t1 = time.perf_counter()
        embedding = np.asarray(
            self._embedder.extract_from_preprocessed(image, for_query=for_query),
            dtype=np.float32,
        )
        dinov2_elapsed = time.perf_counter() - t1

        t2 = time.perf_counter()
        (
            color_hist,
            texture_hist,
            edge_hist,
            pattern_features,
            dominant,
        ) = self.extract_descriptors_from_preprocessed(image)
        descriptors_elapsed = time.perf_counter() - t2

        self._last_timings = ExtractTimings(
            preprocessing=0.0,
            dinov2=dinov2_elapsed,
            descriptors=descriptors_elapsed,
            total=time.perf_counter() - total_start,
        )

        return TileFeatures(
            embedding=embedding,
            color_histogram=color_hist,
            texture_histogram=texture_hist,
            edge_histogram=edge_hist,
            pattern_features=pattern_features,
            dominant_color=dominant,
            width=image.width,
            height=image.height,
        )

    def extract_batch(
        self,
        image_paths: List[str],
        *,
        preprocess_workers: int | None = None,
    ) -> List[TileFeatures]:
        """Extract features for multiple image paths."""
        if not image_paths:
            return []

        if len(image_paths) == 1:
            return [self.extract(image_paths[0])]

        total_start = time.perf_counter()

        t0 = time.perf_counter()
        workers = preprocess_workers or self._preprocess_workers
        worker_count = min(max(1, workers), len(image_paths))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            processed_images = list(pool.map(ImagePreprocessor.preprocess, image_paths))
        preprocess_elapsed = time.perf_counter() - t0

        t1 = time.perf_counter()
        embeddings = self._embedder.extract_batch_from_preprocessed(
            processed_images
        )
        dinov2_elapsed = time.perf_counter() - t1

        t2 = time.perf_counter()
        features_list: List[TileFeatures] = []

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            descriptor_results = list(
                pool.map(self.extract_descriptors_from_preprocessed, processed_images)
            )

        for processed, embedding, descriptor_tuple in zip(
            processed_images,
            embeddings,
            descriptor_results,
        ):
            (
                color_hist,
                texture_hist,
                edge_hist,
                pattern_features,
                dominant,
            ) = descriptor_tuple

            features_list.append(
                TileFeatures(
                    embedding=np.asarray(embedding, dtype=np.float32),
                    color_histogram=color_hist,
                    texture_histogram=texture_hist,
                    edge_histogram=edge_hist,
                    pattern_features=pattern_features,
                    dominant_color=dominant,
                    width=processed.width,
                    height=processed.height,
                )
            )

        descriptors_elapsed = time.perf_counter() - t2
        batch_size = len(image_paths)

        self._last_timings = ExtractTimings(
            preprocessing=preprocess_elapsed / batch_size,
            dinov2=dinov2_elapsed / batch_size,
            descriptors=descriptors_elapsed / batch_size,
            total=(time.perf_counter() - total_start) / batch_size,
        )

        logger.debug(
            "Batch feature extract: count=%d preprocessing=%.3fs dinov2=%.3fs "
            "descriptors=%.3fs",
            batch_size,
            preprocess_elapsed,
            dinov2_elapsed,
            descriptors_elapsed,
        )

        return features_list

    def extract(
        self,
        image_path: str,
        *,
        for_query: bool = False,
    ) -> TileFeatures:

        logger.debug(
            "Extracting AI features: %s (for_query=%s)",
            image_path,
            for_query,
        )

        total_start = time.perf_counter()

        t0 = time.perf_counter()
        views: List[PreprocessedImage] = []
        if for_query:
            views = ImagePreprocessor.prepare_query_views(image_path, max_views=3)
            image = views[0]
        else:
            image = ImagePreprocessor.preprocess(image_path)
        preprocess_elapsed = time.perf_counter() - t0

        if for_query and len(views) > 1:
            features = self._extract_multi_view_query(views)
        else:
            features = self.extract_from_preprocessed(image, for_query=for_query)
        features_elapsed = time.perf_counter() - total_start

        self._last_timings = ExtractTimings(
            preprocessing=preprocess_elapsed,
            dinov2=self._last_timings.dinov2,
            descriptors=self._last_timings.descriptors,
            total=features_elapsed,
        )

        logger.debug(
            "Feature extract timing: preprocessing=%.3fs dinov2=%.3fs "
            "descriptors=%.3fs total=%.3fs views=%d",
            preprocess_elapsed,
            self._last_timings.dinov2,
            self._last_timings.descriptors,
            self._last_timings.total,
            max(1, len(views)),
        )

        return features

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        av = np.asarray(a, dtype=np.float32).ravel()
        bv = np.asarray(b, dtype=np.float32).ravel()
        return float(
            np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-8)
        )

    def _embed_index_view(
        self,
        view: Image.Image,
        *,
        original_size: tuple[int, int],
    ) -> np.ndarray:
        """Letterbox + DINOv2 for an index-time aux crop (same path as primary)."""
        preprocessed = self._finalize_index_pil(view, original_size=original_size)
        return np.asarray(
            self._embedder.extract_from_preprocessed(preprocessed, for_query=False),
            dtype=np.float32,
        )

    @staticmethod
    def _finalize_index_pil(
        view: Image.Image,
        *,
        original_size: tuple[int, int],
        match_pad_to_content: bool = False,
    ) -> PreprocessedImage:
        """Normalize + letterbox an index crop (shared with show_index_crop)."""
        return finalize_index_pil(
            view,
            original_size=original_size,
            match_pad_to_content=match_pad_to_content,
        )

    # Skip near-duplicates of primary. 0.97 was too strict for center-50 on
    # low-contrast marble (measured primary↔center-50 ≈ 0.957 still helps
    # deep crops at ~0.95 vs primary ~0.87).
    _AUX_PRIMARY_MAX_SIM = 0.985
    _AUX_PAIR_MAX_SIM = 0.99

    def _maybe_append_aux(
        self,
        aux: list[np.ndarray],
        primary: np.ndarray,
        candidate: np.ndarray,
        *,
        label: str,
        image_name: str,
    ) -> None:
        """Keep aux vectors that add retrieval coverage (not near-duplicates)."""
        sim_primary = self._cosine_sim(primary, candidate)
        if sim_primary >= self._AUX_PRIMARY_MAX_SIM:
            return
        for existing in aux:
            if self._cosine_sim(existing, candidate) >= self._AUX_PAIR_MAX_SIM:
                return
        aux.append(candidate)
        logger.info(
            "Index aux %s vector for %s (cos_vs_primary=%.3f)",
            label,
            image_name,
            sim_primary,
        )

    def extract_index_vectors(
        self,
        image_path: str,
    ) -> tuple[TileFeatures, list[np.ndarray]]:
        """
        Index-time extract: primary TileFeatures plus optional aux FAISS vectors.

        When ``left_panel_beneficial`` (catalog marketing sheet), the *primary*
        TileFeatures row — embedding **and** color/texture/edge/pattern /
        dominant-color — is taken from ``primary_texture_panel()``, not the
        raw full sheet. Full-sheet layout used to pollute hybrid descriptors
        (measured color similarity 0.075 for xx.jpg vs PGYS2319) while only
        the aux FAISS embedding was clean.

        Aux FAISS vectors still include full-sheet (for sheet self-hit),
        panel_center, and adaptive when beneficial. Near-duplicates are dropped.
        """
        from src.ai.search_quality.views import IndexViewType

        image_name = Path(image_path).name
        aux: list[np.ndarray] = []

        try:
            # Shared with show_index_crops — do not reconstruct crops in parallel.
            prep = prepare_index_primary(image_path)
            raw = prep.raw
            analysis = prep.analysis
            views = list(prep.views)
            panel_pil = prep.panel
            logger.info(
                "Index view plan for %s: kind=%s views=%s "
                "panel=%s center=%s",
                image_name,
                analysis.kind.value,
                [v.view_type.value for v in views],
                analysis.left_panel_beneficial,
                analysis.center_crop_beneficial,
            )

            if prep.primary_source == "panel" and panel_pil is not None:
                primary_pre = prep.primary
                features = self.extract_from_preprocessed(
                    primary_pre,
                    for_query=False,
                )
                logger.info(
                    "Index primary from isolated panel for %s "
                    "(%sx%s → letterbox %sx%s)",
                    image_name,
                    panel_pil.size[0],
                    panel_pil.size[1],
                    primary_pre.pil.size[0],
                    primary_pre.pil.size[1],
                )
            else:
                features = self.extract(image_path, for_query=False)

            primary = np.asarray(features.embedding, dtype=np.float32).ravel()

            for view in views:
                if view.view_type == IndexViewType.PRIMARY:
                    if panel_pil is not None:
                        # Former full-sheet primary → aux for sheet self-hit.
                        emb = self._embed_index_view(
                            view.image,
                            original_size=raw.size,
                        )
                        self._maybe_append_aux(
                            aux,
                            primary,
                            emb,
                            label="full_sheet",
                            image_name=image_name,
                        )
                    continue
                if view.view_type == IndexViewType.PANEL and panel_pil is not None:
                    # Panel crop is already the primary — skip near-dup aux.
                    continue
                emb = self._embed_index_view(view.image, original_size=raw.size)
                self._maybe_append_aux(
                    aux,
                    primary,
                    emb,
                    label=view.view_type.value,
                    image_name=image_name,
                )
        except Exception as exc:
            logger.warning(
                "Index panel/aux path failed for %s (%s) — falling back to "
                "full-sheet primary only",
                image_path,
                exc,
            )
            features = self.extract(image_path, for_query=False)
            aux = []

        return features, aux

    def extract_index_vectors_batch(
        self,
        image_paths: List[str],
    ) -> List[tuple[TileFeatures, list[np.ndarray]]]:
        """
        Batch index-time extract for a folder-scan flush.

        Same ``prepare_index_primary`` + multi-scale DINOv2 fusion + aux
        FAISS vectors as ``extract_index_vectors``, but runs one (chunked)
        batched DINOv2 forward across the whole flush instead of serial
        per-image / per-view calls.
        """
        from src.ai.search_quality.views import IndexViewType

        if not image_paths:
            return []
        if len(image_paths) == 1:
            return [self.extract_index_vectors(image_paths[0])]

        total_start = time.perf_counter()
        t0 = time.perf_counter()

        def _safe_prep(path: str):
            try:
                return prepare_index_primary(path), None
            except Exception as exc:
                return None, exc

        workers = min(max(1, self._preprocess_workers), len(image_paths))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            prep_results = list(pool.map(_safe_prep, image_paths))
        preprocess_elapsed = time.perf_counter() - t0

        primaries: list[PreprocessedImage | None] = []
        for path, (prep, err) in zip(image_paths, prep_results):
            if prep is None:
                logger.warning(
                    "Index prepare failed for %s (%s) — will full-sheet fallback",
                    path,
                    err,
                )
                primaries.append(None)
            else:
                primaries.append(prep.primary)

        valid_indices = [i for i, primary in enumerate(primaries) if primary is not None]
        valid_primaries = [primaries[i] for i in valid_indices]

        t1 = time.perf_counter()
        emb_by_index: dict[int, np.ndarray] = {}
        if valid_primaries:
            batch_fn = getattr(
                self._embedder, "extract_batch_from_preprocessed", None
            )
            if batch_fn is not None:
                batch_embs = batch_fn(valid_primaries)
            else:
                batch_embs = [
                    self._embedder.extract_from_preprocessed(p, for_query=False)
                    for p in valid_primaries
                ]
            for idx, emb in zip(valid_indices, batch_embs):
                emb_by_index[idx] = np.asarray(emb, dtype=np.float32).ravel()
        dinov2_elapsed = time.perf_counter() - t1

        t2 = time.perf_counter()
        desc_by_index: dict[int, tuple] = {}
        if valid_primaries:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                desc_list = list(
                    pool.map(
                        self.extract_descriptors_from_preprocessed,
                        valid_primaries,
                    )
                )
            for idx, desc in zip(valid_indices, desc_list):
                desc_by_index[idx] = desc
        descriptors_elapsed = time.perf_counter() - t2

        # Collect aux crops (letterboxed) across the batch, then one batched
        # DINOv2 pass — same multi-scale fusion as single-file indexing.
        aux_jobs: list[tuple[int, str, PreprocessedImage]] = []
        for i, (path, (prep, _err)) in enumerate(zip(image_paths, prep_results)):
            if prep is None:
                continue
            image_name = Path(path).name
            panel_pil = prep.panel
            raw = prep.raw
            primary = emb_by_index.get(i)
            if primary is None:
                continue
            for view in prep.views:
                if view.view_type == IndexViewType.PRIMARY:
                    if panel_pil is not None:
                        aux_jobs.append(
                            (
                                i,
                                "full_sheet",
                                self._finalize_index_pil(
                                    view.image, original_size=raw.size
                                ),
                            )
                        )
                    continue
                if view.view_type == IndexViewType.PANEL and panel_pil is not None:
                    continue
                aux_jobs.append(
                    (
                        i,
                        view.view_type.value,
                        self._finalize_index_pil(
                            view.image, original_size=raw.size
                        ),
                    )
                )

        aux_by_index: dict[int, list[np.ndarray]] = {i: [] for i in range(len(image_paths))}
        if aux_jobs:
            t_aux = time.perf_counter()
            aux_pre = [job[2] for job in aux_jobs]
            batch_fn = getattr(
                self._embedder, "extract_batch_from_preprocessed", None
            )
            if batch_fn is not None:
                aux_embs = batch_fn(aux_pre)
            else:
                aux_embs = [
                    self._embedder.extract_from_preprocessed(p, for_query=False)
                    for p in aux_pre
                ]
            dinov2_elapsed += time.perf_counter() - t_aux
            for (img_idx, label, _pre), emb in zip(aux_jobs, aux_embs):
                image_name = Path(image_paths[img_idx]).name
                self._maybe_append_aux(
                    aux_by_index[img_idx],
                    emb_by_index[img_idx],
                    np.asarray(emb, dtype=np.float32).ravel(),
                    label=label,
                    image_name=image_name,
                )

        results: list[tuple[TileFeatures, list[np.ndarray]]] = []
        for i, path in enumerate(image_paths):
            if i not in emb_by_index or i not in desc_by_index:
                features, aux = self.extract_index_vectors(path)
                results.append((features, aux))
                continue
            (
                color_hist,
                texture_hist,
                edge_hist,
                pattern_features,
                dominant,
            ) = desc_by_index[i]
            primary_pre = primaries[i]
            assert primary_pre is not None
            features = TileFeatures(
                embedding=emb_by_index[i],
                color_histogram=color_hist,
                texture_histogram=texture_hist,
                edge_histogram=edge_hist,
                pattern_features=pattern_features,
                dominant_color=dominant,
                width=primary_pre.width,
                height=primary_pre.height,
            )
            results.append((features, aux_by_index.get(i, [])))

        n = len(image_paths)
        self._last_timings = ExtractTimings(
            preprocessing=preprocess_elapsed / n,
            dinov2=dinov2_elapsed / n,
            descriptors=descriptors_elapsed / n,
            total=(time.perf_counter() - total_start) / n,
        )
        logger.info(
            "Index batch extract: count=%d preprocess=%.3fs dinov2=%.3fs "
            "descriptors=%.3fs",
            n,
            preprocess_elapsed,
            dinov2_elapsed,
            descriptors_elapsed,
        )
        return results

    def extract_for_search(
        self,
        image_path: str,
        *,
        preloaded: Image.Image | None = None,
        query_origin: str | None = None,
    ) -> tuple[TileFeatures, list[np.ndarray]]:
        """
        Query-only adaptive extract (index unchanged).

        Room / phone / partial-crop queries: isolate or complementary
        multi-crop (Query Analyzer). Crop-tool outputs (Auto / Precise /
        Manual) skip re-isolation and perspective straighten — they are
        already a tile surface. All other queries: classic
        ``preprocess_for_query`` single embedding (v1.2.31 path).

        FAISS merges crops by MAX per tile_id in SearchTilesUseCase.
        """
        total_start = time.perf_counter()
        t0 = time.perf_counter()
        path = Path(image_path)
        image = ImagePreprocessor.to_rgb(
            preloaded if preloaded is not None else ImagePreprocessor.load(path)
        )

        from src.ai.search_quality.query_analyzer import QueryKind, analyze_query
        from src.ai.search_quality.query_origin import QueryOrigin, resolve_query_origin
        from src.ai.search_quality.query_views import (
            collect_crop_tool_pils,
            collect_query_crop_pils,
        )

        analysis = analyze_query(image)
        origin = resolve_query_origin(path, query_origin)
        use_crop_tool_views = origin is QueryOrigin.CROP_TOOL
        use_multi = (
            not use_crop_tool_views
            and analysis.kind
            in {
                QueryKind.ROOM_SCENE,
                QueryKind.PHONE_SCREENSHOT,
                QueryKind.PARTIAL_CROP,
            }
            and "tilevision_crops" not in path.as_posix().lower()
        )

        original_width, original_height = image.size
        if use_crop_tool_views:
            max_cap = ImagePreprocessor._capped_query_max_views(2)
            crop_pils = collect_crop_tool_pils(image, max_views_cap=max_cap)
            views = [
                ImagePreprocessor._finalize_query_pil(
                    crop,
                    original_width=original_width,
                    original_height=original_height,
                    straighten=False,
                )
                for crop in crop_pils
            ]
        elif use_multi:
            max_cap = ImagePreprocessor._capped_query_max_views(3)
            _, crop_pils = collect_query_crop_pils(
                image, analysis=analysis, max_views_cap=max_cap
            )
            views = [
                ImagePreprocessor._finalize_query_pil(
                    crop,
                    original_width=original_width,
                    original_height=original_height,
                )
                for crop in crop_pils
            ]
        else:
            # Preserve v1.2.31 single-pass behavior for clean/partial/catalogue
            # (rotation, crops, originals). Only the sheet-vs-room gate changed.
            views = ImagePreprocessor.prepare_query_views(
                path,
                max_views=1,
                preloaded=image,
            )

        preprocess_elapsed = time.perf_counter() - t0

        t1 = time.perf_counter()
        batch_fn = getattr(self._embedder, "extract_query_views_batch", None)
        if batch_fn is not None and len(views) > 1:
            embeddings = batch_fn(views)
        else:
            embeddings = []
            for view in views:
                embeddings.append(
                    np.asarray(
                        self._embedder.extract_from_preprocessed(
                            view, for_query=True
                        ),
                        dtype=np.float32,
                    )
                )
        dinov2_elapsed = time.perf_counter() - t1

        features = self._fuse_query_embeddings(
            views[0], embeddings, dinov2_elapsed
        )
        self._last_timings = ExtractTimings(
            preprocessing=preprocess_elapsed,
            dinov2=dinov2_elapsed,
            descriptors=self._last_timings.descriptors,
            total=time.perf_counter() - total_start,
        )
        logger.info(
            "Search extract (adaptive query): kind=%s origin=%s "
            "embed_views=%d letterbox=%s preprocess=%.2fs dinov2=%.2fs total=%.2fs",
            analysis.kind.value,
            origin.value,
            len(embeddings),
            [v.pil.size for v in views],
            preprocess_elapsed,
            dinov2_elapsed,
            self._last_timings.total,
        )
        return features, embeddings

    def _extract_multi_view_query(
        self,
        views: List[PreprocessedImage],
    ) -> TileFeatures:
        """
        Embed several query crops and fuse DINOv2 vectors (L2-normalized mean).

        Descriptors come from the primary (best) crop. Query-only — does not
        change indexed catalog vectors.
        """
        embeddings: list[np.ndarray] = []
        dinov2_elapsed = 0.0

        for view in views:
            t1 = time.perf_counter()
            emb = np.asarray(
                self._embedder.extract_from_preprocessed(view, for_query=True),
                dtype=np.float32,
            )
            dinov2_elapsed += time.perf_counter() - t1
            embeddings.append(emb)

        return self._fuse_query_embeddings(views[0], embeddings, dinov2_elapsed)

    def _fuse_query_embeddings(
        self,
        primary: PreprocessedImage,
        embeddings: list[np.ndarray],
        dinov2_elapsed: float,
    ) -> TileFeatures:
        total_start = time.perf_counter()
        stacked = np.vstack(embeddings)
        fused = stacked.mean(axis=0)
        fused = fused / (np.linalg.norm(fused) + 1e-8)
        fused = fused.astype(np.float32)

        t2 = time.perf_counter()
        (
            color_hist,
            texture_hist,
            edge_hist,
            pattern_features,
            dominant,
        ) = self.extract_descriptors_from_preprocessed(primary)
        descriptors_elapsed = time.perf_counter() - t2

        self._last_timings = ExtractTimings(
            preprocessing=0.0,
            dinov2=dinov2_elapsed,
            descriptors=descriptors_elapsed,
            total=time.perf_counter() - total_start + dinov2_elapsed,
        )

        logger.info(
            "Query multi-crop DINOv2 fuse: input_views=%d dim=%d",
            len(embeddings),
            fused.shape[0],
        )

        return TileFeatures(
            embedding=fused,
            color_histogram=color_hist,
            texture_histogram=texture_hist,
            edge_histogram=edge_hist,
            pattern_features=pattern_features,
            dominant_color=dominant,
            width=primary.width,
            height=primary.height,
        )
