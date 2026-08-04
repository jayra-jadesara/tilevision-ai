"""
Visual similarity search use case module for TileVision AI.

Given a query image, extracts features, performs FAISS vector search, and merges
matching items with SQLite database metadata and cached thumbnail paths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from src.ai.pattern_classifier import PatternClassifier
from src.ai.query_cache import QUERY_EMBEDDING_CACHE
from src.ai.similarity_score import calibrate_display_percent
from src.ai.inference_guard import InferenceBusyError
from src.ai.models import TileFeatures

from src.core.models import TileImage, SearchResult
from src.data.repository_interface import IImageRepository
from src.ai.feature_extractor import FeatureExtractor
from src.ai.reranker import HybridReRanker
from src.ai.vector_index import FaissIndexManager
from src.utils.image_utils import (
    compute_sha256,
    compute_dhash,
    compute_dhash_from_image,
    hamming_distance,
    get_thumbnail_path,
)
from src.utils.pipeline_timing import PipelineTimer
from src.utils.search_stages import (
    STAGE_EMBEDDING_CACHE_HIT,
    STAGE_EMBEDDING_GENERATED,
    STAGE_EMBEDDING_NORMALIZED,
    STAGE_FAISS_SEARCH,
    STAGE_HEALTH_OK,
    STAGE_IMAGE_DECODED,
    STAGE_PREPROCESS_COMPLETE,
    STAGE_RERANK_COMPLETE,
    STAGE_RESULTS_READY,
    STAGE_SQLITE_HYDRATE,
    STAGE_THUMBNAILS_QUEUED,
    STAGE_WEAK_FILTER,
    log_search_failure,
    log_search_stage,
)
from src.ai.preprocess.image_preprocessor import ImagePreprocessor

logger = logging.getLogger("tilevision.core.use_cases.search_tiles")

# Filter fields supported by execute()'s `filters` parameter (Feature 8).
# Matches SQLiteImageRepository._DISTINCT_VALUE_ALLOWED_FIELDS — kept as a
# separate allow-list here since this is the boundary that receives
# caller/UI-supplied filter keys directly.
_ALLOWED_FILTER_FIELDS = frozenset({"brand", "category", "color", "size"})

# When filters are active, FAISS is queried for a wider candidate pool than
# top_k (since some candidates will get filtered out), then narrowed back
# down to top_k after matching metadata. Capped to avoid a pathological
# widen-forever cost on a catalog with a very restrictive filter.
_FILTER_CANDIDATE_MULTIPLIER = 10
_FILTER_CANDIDATE_CAP = 2000
# When metadata filters are active, rerank the full filtered ID set directly
# (instead of FAISS top-K only) up to this many tiles for accurate results.
_FILTERED_FULL_RERANK_CAP = 2000

# Unfiltered FAISS retrieval pool bounds (Phase 7).
_FAISS_CANDIDATE_MIN = 50
_FAISS_CANDIDATE_MAX = 200

# Perceptual hashes within this Hamming distance are treated as near-exact
# self matches (same tile, different compression/crop).
_NEAR_EXACT_DHASH_THRESHOLD = 3

# Drop clearly weak matches so a small catalog does not always fill top_k
# with unrelated tiles (room photos, marble, etc.).
_WEAK_RESULT_RELATIVE_FLOOR = 0.60
_WEAK_RESULT_ABSOLUTE_RAW_FLOOR = 0.38

# Crop-from-catalog: when embedding similarity to the source product is this
# high, treat it as the same catalog tile (100% match).
_CROP_SOURCE_EMBEDDING_THRESHOLD = 0.78

# When FAISS retrieves a tile via an aux texture-panel vector, FlatIP cosine
# can be ≫ hybrid.embedding (which always uses the layout-heavy primary).
# Trust the stronger FAISS hit so parent sheets stay in Top-5 for slab crops.
_FAISS_AUX_BOOST_MIN = 0.80
_FAISS_AUX_BOOST_GAP = 0.12
# Strong aux hit = this IS the parent product for a dropped texture crop.
# Rank it above an indexed copy of the crop file itself (same-path 100%).
_FAISS_PARENT_SHEET_TOP_MIN = 0.88
_QUERY_SELF_MATCH_SCORE = 0.97


class SearchTilesUseCase:
    """
    Use case to query visual similarity of a tile sample against the indexed catalog.
    """

    def __init__(
        self,
        image_repository: IImageRepository,
        feature_extractor: FeatureExtractor,
        vector_index: FaissIndexManager,
        thumbnail_dir: str,
    ) -> None:
        """
        Initialize the search use case.

        Args:
            image_repository: Repository interface for SQLite.
            embedder: CLIP model embedder wrapper.
            vector_index: FAISS index manager wrapper.
            thumbnail_dir: Folder path where thumbnails are cached.
        """
        self._repo = image_repository
        self._feature_extractor = feature_extractor
        self._index = vector_index
        self._thumbnail_dir = Path(thumbnail_dir)

        self._reranker = HybridReRanker()

    def get_filter_options(self) -> Dict[str, List[str]]:
        """
        Retrieve the available values for each filterable metadata field,
        for populating filter dropdowns in the Search view.

        Returns:
            Dict mapping field name -> sorted list of distinct values
            currently present in the catalog (e.g. {"brand": ["Kajaria", ...]}).
        """
        return {
            field: self._repo.get_distinct_values(field)
            for field in sorted(_ALLOWED_FILTER_FIELDS)
        }

    @staticmethod
    def _compute_faiss_search_k(top_k: int, total_vectors: int) -> int:
        """Return the unfiltered FAISS candidate pool size (Phase 7: 50–200).

        Over-fetch by 2× so multi-vector tile ids (sheet + texture panel) do
        not shrink unique-id recall below the historical target.
        """
        unique_target = min(
            max(top_k * 5, _FAISS_CANDIDATE_MIN),
            _FAISS_CANDIDATE_MAX,
            total_vectors,
        )
        return min(unique_target * 2, total_vectors)

    def _search_faiss_multi_crop(
        self,
        embeddings: list,
        search_k: int,
    ) -> tuple[List[int], dict[int, float]]:
        """
        Run FAISS for each query crop and merge by best similarity per tile id.

        Returns ordered unique ids plus the best FAISS Inner-Product (cosine)
        per id. Aux texture-panel vectors share a tile id with the primary
        sheet embedding; the best score must flow into rerank or layout-heavy
        primaries stay near ~27% after hybrid scoring.
        """
        if not embeddings:
            return [], {}

        best_score: dict[int, float] = {}
        for emb in embeddings:
            ids, scores = self._index.search_vectors(emb, search_k)
            for tile_id, score in zip(ids, scores):
                prev = best_score.get(tile_id)
                if prev is None or float(score) > prev:
                    best_score[tile_id] = float(score)

        ordered = sorted(best_score.items(), key=lambda item: item[1], reverse=True)
        matching_ids = [tile_id for tile_id, _score in ordered]
        logger.info(
            "Multi-crop FAISS merge: crops=%d unique_ids=%d",
            len(embeddings),
            len(matching_ids),
        )
        return matching_ids, best_score

    def get_index_health(self):
        """Return feature-version compatibility status for the indexed catalog."""
        return self._repo.get_feature_version_status()

    def get_searchable_count(self) -> int:
        """Return how many vectors FAISS can actually search right now."""
        try:
            return int(self._index.get_total_count())
        except InferenceBusyError:
            # Caller must treat busy separately from an empty index.
            raise
        except Exception as exc:
            logger.warning("Could not read FAISS searchable count: %s", exc)
            raise RuntimeError(
                f"Could not read the searchable vector index: {exc}"
            ) from exc

    def execute(
        self,
        query_image_path: str,
        top_k: int = 20,
        filters: Optional[Dict[str, str]] = None,
        on_stage: Optional[Callable[[str], None]] = None,
    ) -> List[SearchResult]:
        """
        Execute visual similarity search for a query tile image.

        Args:
            query_image_path: Absolute path to the user's target search image.
            top_k: Maximum number of closest matches to return.
            filters: Optional dict of metadata field -> required value
                (e.g. {"brand": "Kajaria", "category": "Floor"}). Only
                results matching ALL provided filters are returned. Keys
                must be in _ALLOWED_FILTER_FIELDS; unknown keys are ignored
                (not treated as an error, since a UI might pass a superset
                of possible filter widgets where some are left at "Any").
            on_stage: Optional callback(str) for UI progress breadcrumbs.

        Returns:
            A list of SearchResult objects sorted by similarity score descending.
        """
        query_path = Path(query_image_path)
        if not query_path.exists() or not query_path.is_file():
            raise FileNotFoundError(f"Query image does not exist: {query_image_path}")

        top_k = max(1, int(top_k))
        active_filters = {
            k: v for k, v in (filters or {}).items()
            if k in _ALLOWED_FILTER_FIELDS and v
        }

        def stage(name: str, detail: str = "") -> None:
            log_search_stage(logger, name, detail=detail, on_stage=on_stage)

        logger.info(
            f"Initiating similarity search query for: {query_path.name} "
            f"(top_k={top_k}, filters={active_filters or 'none'})"
        )

        version_status = self._repo.get_feature_version_status()
        if not version_status.is_compatible and version_status.stale_count > 0:
            raise RuntimeError(
                "Indexed features are outdated. "
                f"{version_status.stale_count} of {version_status.indexed_count} "
                "tiles need re-indexing. Use Settings → Rebuild FAISS Index."
            )

        try:
            # Ensure DINOv2 weights are loaded before decode/embed.
            if hasattr(self._feature_extractor, "load_model"):
                self._feature_extractor.load_model()

            # Ensure FAISS is loaded and dimension matches production embeddings.
            if getattr(self._index, "_index", None) is None:
                self._index.load_index()
            index_dim = int(self._index.embedding_dimension())
            from src.ai.feature_versions import CURRENT_EMBEDDING_DIMENSION

            if index_dim != CURRENT_EMBEDDING_DIMENSION:
                raise RuntimeError(
                    f"FAISS embedding dimension {index_dim} does not match "
                    f"DINOv2 dimension {CURRENT_EMBEDDING_DIMENSION}. "
                    "Rebuild FAISS Index."
                )
            stage(STAGE_HEALTH_OK, f"faiss_dim={index_dim}")

            timer = PipelineTimer("SEARCH TIMING")

            # ── 1. Resolve query features (memory cache → catalog → embed) ──
            query_features: TileFeatures | None = None
            query_embeddings: list = []
            query_sha256 = ""
            query_dhash = ""
            cache_status = "miss"
            preloaded_image = None

            cached_query = QUERY_EMBEDDING_CACHE.get(query_path)
            if cached_query is not None:
                query_features = cached_query.features
                query_embeddings = [
                    np.asarray(e, dtype=np.float32) for e in cached_query.embeddings
                ]
                cache_status = "hit"
                timer.timings.record("image_load", 0.0)
                timer.timings.record("crop", 0.0)
                timer.timings.record("embedding", 0.0)
                timer.timings.record("descriptors", 0.0)
                stage(STAGE_EMBEDDING_CACHE_HIT, query_path.name)
                logger.info(
                    "Reusing cached query embedding: %s",
                    query_path.name,
                )

            if query_features is None:
                # Cheap byte hash first — may hit the catalog without decoding.
                with timer.measure("image_load"):
                    query_sha256 = compute_sha256(query_path)

                cached_tile = self._repo.get_by_path(str(query_path.resolve()))
                if (
                    cached_tile
                    and cached_tile.is_indexed
                    and cached_tile.features is not None
                    and cached_tile.sha256_hash == query_sha256
                ):
                    query_features = cached_tile.features
                    query_embeddings = [query_features.embedding]
                    cache_status = "catalog"
                    timer.timings.record("crop", 0.0)
                    timer.timings.record("embedding", 0.0)
                    timer.timings.record("descriptors", 0.0)
                    stage(STAGE_EMBEDDING_CACHE_HIT, f"catalog:{query_path.name}")
                    logger.info(
                        "Reusing indexed features for catalog query: %s",
                        query_path.name,
                    )

            if query_features is None:
                # Decode the query image exactly once for dHash + embedding.
                with timer.measure("image_load"):
                    try:
                        preloaded_image = ImagePreprocessor.load(query_path)
                    except Exception as exc:
                        log_search_failure(logger, STAGE_IMAGE_DECODED, exc)
                        raise ValueError(
                            f"Selected file is not a valid, readable image: "
                            f"{query_path.name}"
                        ) from exc
                    if not query_sha256:
                        query_sha256 = compute_sha256(query_path)
                    query_dhash = compute_dhash_from_image(preloaded_image)
                stage(STAGE_IMAGE_DECODED, query_path.name)

                logger.info("Computing embedding for query image...")
                extract_for_search = getattr(
                    self._feature_extractor, "extract_for_search", None
                )
                if extract_for_search is not None:
                    query_features, query_embeddings = extract_for_search(
                        str(query_path),
                        preloaded=preloaded_image,
                    )
                else:
                    query_features = self._feature_extractor.extract(
                        str(query_path),
                        for_query=True,
                    )
                    query_embeddings = [query_features.embedding]
                if query_features is None or not query_embeddings:
                    raise RuntimeError(
                        "DINOv2 embedding generation returned no vectors for the query image."
                    )
                stage(STAGE_PREPROCESS_COMPLETE)
                stage(
                    STAGE_EMBEDDING_GENERATED,
                    f"crops={len(query_embeddings)} dim={len(query_features.embedding)}",
                )
                extract_timings = self._feature_extractor.last_timings
                # Map internal extract stages onto the required profile labels.
                timer.timings.record("crop", extract_timings.preprocessing)
                timer.timings.record("embedding", extract_timings.dinov2)
                timer.timings.record("descriptors", extract_timings.descriptors)
                QUERY_EMBEDDING_CACHE.put(
                    query_path,
                    query_features,
                    query_embeddings,
                )

            # Exact-match SHA for memory-cache hits (no decode required).
            if not query_sha256:
                with timer.measure("image_load"):
                    query_sha256 = compute_sha256(query_path)

            # Normalize sanity — FAISS search also L2-normalizes; log explicitly.
            emb0 = np.asarray(
                query_embeddings[0] if query_embeddings else query_features.embedding,
                dtype=np.float32,
            )
            norm = float(np.linalg.norm(emb0))
            if not np.isfinite(norm) or norm < 1e-8:
                raise RuntimeError(
                    "Query embedding is empty or not finite after DINOv2 — cannot search."
                )
            stage(STAGE_EMBEDDING_NORMALIZED, f"norm={norm:.4f}")

            # ----------------------------------------
            # Detect query pattern type
            # ----------------------------------------

            query_pattern_type = PatternClassifier.classify(
                query_features
            )

            logger.info(
                "Query pattern type detected: %s",
                query_pattern_type.value,
            )

            # 2. Retrieve candidate tiles (FAISS or metadata-filtered full set).
            total_vectors = self._index.get_total_count()
            timer.set_meta(
                cache=cache_status,
                catalog_size=total_vectors,
                faiss_index=self._index.index_type_name(),
                embedding_dim=self._index.embedding_dimension(),
            )

            if total_vectors <= 0:
                raise RuntimeError(
                    "The searchable vector index is empty even though tiles may be listed. "
                    "Go to Settings → Rebuild FAISS Index, then search again."
                )

            filtered_ids: Optional[set[int]] = None
            if active_filters:
                filtered_ids = set(self._repo.get_ids_matching_filters(active_filters))
                if not filtered_ids:
                    logger.warning(
                        "No indexed tiles match metadata filters: %s",
                        active_filters,
                    )
                    raise RuntimeError(
                        "No indexed tiles match the active search filters "
                        f"({active_filters}).\n\n"
                        "Clear Brand/Category/Color/Size filters (set to Any), "
                        "then drop your image again."
                    )

            candidates: List[TileImage] = []
            matching_ids: List[int] = []
            faiss_scores: dict[int, float] = {}

            if filtered_ids is not None and len(filtered_ids) <= _FILTERED_FULL_RERANK_CAP:
                logger.info(
                    "Metadata filters active — reranking %d filtered catalog tile(s) directly.",
                    len(filtered_ids),
                )
                with timer.measure("metadata"):
                    matched_tiles = self._repo.get_by_ids(list(filtered_ids))
                candidates = [
                    tile for tile in matched_tiles if tile.features is not None
                ]
                stage(
                    STAGE_SQLITE_HYDRATE,
                    f"{len(candidates)} feature-ready of {len(matched_tiles)} filtered",
                )
            else:
                search_k = self._compute_faiss_search_k(top_k, total_vectors)

                if active_filters:
                    search_k = min(
                        max(top_k * _FILTER_CANDIDATE_MULTIPLIER, top_k),
                        _FILTER_CANDIDATE_CAP,
                        total_vectors,
                    )

                logger.info(
                    "Querying FAISS vector index (search_k=%s, query_crops=%s, cache=%s)...",
                    search_k,
                    len(query_embeddings) or 1,
                    cache_status,
                )
                with timer.measure("faiss"):
                    matching_ids, faiss_scores = self._search_faiss_multi_crop(
                        query_embeddings or [query_features.embedding],
                        search_k,
                    )

                if not matching_ids:
                    stage(STAGE_FAISS_SEARCH, "0 IDs")
                    logger.info("No matching records found in vector index.")
                    return []

                stage(STAGE_FAISS_SEARCH, f"{len(matching_ids)} IDs")
                logger.info(
                    "Retrieving database metadata for matching IDs: %s",
                    matching_ids,
                )
                with timer.measure("metadata"):
                    matched_tiles = self._repo.get_by_ids(matching_ids)

                tile_map = {t.id: t for t in matched_tiles if t.id is not None}
                missing_ids = 0
                missing_features = 0

                for record_id in matching_ids:
                    tile = tile_map.get(record_id)
                    if tile is None:
                        missing_ids += 1
                        continue
                    if filtered_ids is not None and record_id not in filtered_ids:
                        continue
                    if not self._matches_filters(tile, active_filters):
                        continue
                    if tile.features is None:
                        missing_features += 1
                        continue
                    candidates.append(tile)

                stage(
                    STAGE_SQLITE_HYDRATE,
                    f"{len(candidates)} records "
                    f"(missing_ids={missing_ids}, missing_features={missing_features})",
                )

                if matching_ids and not candidates:
                    raise RuntimeError(
                        "FAISS returned similar tiles but none could be loaded from SQLite "
                        f"(missing rows={missing_ids}, incomplete features={missing_features}).\n\n"
                        "The catalogue index is out of sync. Go to Settings → Rebuild FAISS Index, "
                        "then search again."
                    )

            logger.info(
                "Candidates for reranking: %d",
                len(candidates),
            )

            # -------------------------------------------------------
            # Hybrid Re-ranking (color compatibility applied as soft penalty)
            # -------------------------------------------------------

            catalog_source_tile: TileImage | None = None
            crop_stem = self._resolve_crop_source_stem(query_path)
            if crop_stem is not None:
                with timer.measure("metadata"):
                    catalog_source_tile = self._find_catalog_tile_by_stem(crop_stem)
                if catalog_source_tile is not None:
                    logger.info(
                        "Crop search linked to catalog tile: %s",
                        catalog_source_tile.file_name,
                    )

            reranked = []
            query_resolved = str(query_path.resolve())

            with timer.measure("reranking"):
                for tile in candidates:

                    if tile.features is None:
                        continue

                    candidate_pattern_type = PatternClassifier.classify(
                        tile.features
                    )

                    hybrid = self._reranker.score(
                        query_features,
                        tile.features,
                        query_pattern_type=query_pattern_type,
                        candidate_pattern_type=candidate_pattern_type,
                    )

                    exact_match = self._is_exact_match(
                        tile,
                        query_sha256,
                        query_dhash,
                    )
                    same_query_file = False
                    try:
                        same_query_file = (
                            Path(tile.file_path).resolve() == Path(query_resolved)
                        )
                    except OSError:
                        same_query_file = False

                    faiss_cos = float(faiss_scores.get(tile.id, 0.0) or 0.0)
                    # Dropped texture crop of a marketing sheet: aux FAISS hit on
                    # the parent sheet must rank ABOVE the crop file itself when
                    # that crop was also saved into the catalogue folder.
                    if (
                        not same_query_file
                        and faiss_cos >= _FAISS_PARENT_SHEET_TOP_MIN
                        and faiss_cos >= hybrid.embedding + _FAISS_AUX_BOOST_GAP
                    ):
                        exact_match = True
                        final_score = 1.0
                        logger.info(
                            "Parent sheet Top-1 via FAISS aux | %s | faiss=%.3f "
                            "hybrid_emb=%.3f",
                            tile.file_name,
                            faiss_cos,
                            hybrid.embedding,
                        )
                    elif (
                        not exact_match
                        and faiss_cos >= _FAISS_AUX_BOOST_MIN
                        and faiss_cos >= hybrid.embedding + _FAISS_AUX_BOOST_GAP
                    ):
                        boosted = max(
                            0.0,
                            min(1.0, 0.72 * faiss_cos + 0.28 * float(hybrid.final)),
                        )
                        final_score = max(float(hybrid.final), boosted)
                        logger.info(
                            "FAISS aux boost | %s | faiss=%.3f hybrid_emb=%.3f "
                            "hybrid_final=%.3f → final=%.3f",
                            tile.file_name,
                            faiss_cos,
                            hybrid.embedding,
                            hybrid.final,
                            final_score,
                        )
                    elif (
                        not exact_match
                        and catalog_source_tile is not None
                        and tile.id == catalog_source_tile.id
                        and max(hybrid.embedding, faiss_cos)
                        >= _CROP_SOURCE_EMBEDDING_THRESHOLD
                    ):
                        exact_match = True
                        final_score = 1.0
                    elif exact_match and same_query_file:
                        # Keep self-hit visible, but below a parent-sheet aux hit.
                        final_score = _QUERY_SELF_MATCH_SCORE
                        exact_match = False
                    else:
                        final_score = 1.0 if exact_match else hybrid.final

                    logger.debug(
                        "RERANK | %-45s | embedding=%.4f pattern=%.4f "
                        "color=%.4f texture=%.4f edge=%.4f final=%.4f exact=%s",
                        tile.file_name,
                        hybrid.embedding,
                        hybrid.pattern,
                        hybrid.color,
                        hybrid.texture,
                        hybrid.edge,
                        final_score,
                        exact_match,
                    )

                    reranked.append(
                        (
                            final_score,
                            tile,
                            exact_match,
                        )
                    )

            reranked.sort(
                key=lambda item: item[0],
                reverse=True,
            )
            stage(STAGE_RERANK_COMPLETE, f"{len(reranked)} candidates")

            if candidates and not reranked:
                raise RuntimeError(
                    "Candidates were found but hybrid reranking produced no scored tiles. "
                    "Rebuild FAISS Index or re-index the catalogue folder."
                )

            reranked = self._filter_weak_results(reranked, top_k)
            stage(STAGE_WEAK_FILTER, f"kept {len(reranked)}")

            results: List[SearchResult] = []

            # Thumbnail paths only — existence / QPixmap decode is deferred to
            # the UI so search returns as soon as ranking finishes.
            with timer.measure("thumbnail"):
                for score, tile, exact_match in reranked[:top_k]:
                    thumbnail_path = get_thumbnail_path(
                        Path(tile.file_path),
                        self._thumbnail_dir,
                    )
                    similarity_percentage = calibrate_display_percent(
                        score,
                        exact_match=exact_match,
                    )
                    results.append(
                        SearchResult(
                            tile=tile,
                            similarity_score=similarity_percentage,
                            thumbnail_path=str(thumbnail_path),
                        )
                    )

            stage(STAGE_THUMBNAILS_QUEUED, f"{len(results)}")
            stage(STAGE_RESULTS_READY, f"{len(results)} results")
            timer.log_summary(log=logger)
            return results
        except InferenceBusyError:
            raise
        except Exception as e:
            log_search_failure(logger, "execute", e)
            logger.error(f"Failed to execute tile search query: {e}")
            raise RuntimeError(f"Visual similarity search execution error: {e}") from e

    @staticmethod
    def _matches_filters(tile: TileImage, filters: Dict[str, str]) -> bool:
        """Check whether a tile's metadata satisfies every active filter."""
        for field, required_value in filters.items():
            tile_value = getattr(tile, field, None)
            if not tile_value or tile_value.strip().lower() != required_value.strip().lower():
                return False
        return True

    @staticmethod
    def _is_exact_match(
        tile: TileImage,
        query_sha256: str,
        query_dhash: str,
    ) -> bool:
        """Detect byte-identical or near-identical catalog self-matches."""
        if query_sha256 and tile.sha256_hash and query_sha256 == tile.sha256_hash:
            return True

        if query_dhash and tile.perceptual_hash:
            distance = hamming_distance(query_dhash, tile.perceptual_hash)
            if 0 <= distance <= _NEAR_EXACT_DHASH_THRESHOLD:
                return True

        return False

    @staticmethod
    def _resolve_crop_source_stem(query_path: Path) -> Optional[str]:
        """
        Extract the original catalog filename stem from a Crop & Search temp file.

        Example: crop_5mm-white-dotted-ceramic-floor-tile-500x500_12345.jpg
        -> 5mm-white-dotted-ceramic-floor-tile-500x500
        """
        normalized = str(query_path).replace("\\", "/").lower()
        if "tilevision_crops" not in normalized:
            return None

        filename = normalized.rsplit("/", 1)[-1]
        stem = Path(filename).stem
        if not stem.startswith("crop_"):
            return None

        remainder = stem[5:]
        if "_" in remainder:
            base, suffix = remainder.rsplit("_", 1)
            if suffix.isdigit():
                return base
        return remainder

    def _find_catalog_tile_by_stem(self, stem: str) -> Optional[TileImage]:
        """Find an indexed catalog tile whose filename stem matches."""
        target = stem.strip().lower()
        if not target:
            return None

        lookup = getattr(self._repo, "get_indexed_by_file_stem", None)
        if callable(lookup):
            return lookup(target)

        for tile in self._repo.get_all():
            if not tile.is_indexed:
                continue
            if Path(tile.file_name).stem.lower() == target:
                return tile
        return None

    @staticmethod
    def _filter_weak_results(
        reranked: List[tuple[float, TileImage, bool]],
        top_k: int,
    ) -> List[tuple[float, TileImage, bool]]:
        """
        Remove weak tail results so unrelated catalog tiles are not shown
        just to fill top_k in a small showroom database.

        Reliability: never return an empty list when FAISS+rerank produced
        candidates — always keep at least the best match so a valid drop
        cannot silently become "no results".
        """
        if not reranked:
            return []

        reference_score = reranked[0][0]
        if reranked[0][2]:
            for score, _, exact_match in reranked[1:]:
                if not exact_match:
                    reference_score = score
                    break

        min_raw = max(
            reference_score * _WEAK_RESULT_RELATIVE_FLOOR,
            _WEAK_RESULT_ABSOLUTE_RAW_FLOOR,
        )

        kept: List[tuple[float, TileImage, bool]] = []
        for score, tile, exact_match in reranked:
            if exact_match or score >= min_raw:
                kept.append((score, tile, exact_match))
            if len(kept) >= top_k:
                break

        if not kept:
            # Always surface the strongest match rather than an empty UI.
            keep_n = min(max(1, min(3, top_k)), len(reranked))
            kept = list(reranked[:keep_n])
            logger.warning(
                "Weak-result filter would have kept 0 of %d — "
                "retaining top %d match(es) so search is never empty "
                "(best_score=%.3f, min_raw=%.3f)",
                len(reranked),
                keep_n,
                reranked[0][0],
                min_raw,
            )
        else:
            logger.info(
                "Weak-result filter: kept %d of %d candidates (min_raw=%.3f)",
                len(kept),
                len(reranked),
                min_raw,
            )
        return kept
