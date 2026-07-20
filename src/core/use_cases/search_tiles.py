"""
Visual similarity search use case module for TileVision AI.

Given a query image, extracts features, performs FAISS vector search, and merges
matching items with SQLite database metadata and cached thumbnail paths.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.ai.pattern_classifier import PatternClassifier
from src.ai.similarity_score import calibrate_display_percent

from src.core.models import TileImage, SearchResult
from src.data.repository_interface import IImageRepository
from src.ai.feature_extractor import FeatureExtractor
from src.ai.reranker import HybridReRanker
from src.ai.vector_index import FaissIndexManager
from src.utils.image_utils import (
    compute_sha256,
    compute_dhash,
    hamming_distance,
    get_thumbnail_path,
    validate_image,
)
from src.utils.pipeline_timing import PipelineTimer

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
_WEAK_RESULT_RELATIVE_FLOOR = 0.52
_WEAK_RESULT_ABSOLUTE_RAW_FLOOR = 0.30

# Crop-from-catalog: when embedding similarity to the source product is this
# high, treat it as the same catalog tile (100% match).
_CROP_SOURCE_EMBEDDING_THRESHOLD = 0.78


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
        """Return the unfiltered FAISS candidate pool size (Phase 7: 50–200)."""
        return min(
            max(top_k * 5, _FAISS_CANDIDATE_MIN),
            _FAISS_CANDIDATE_MAX,
            total_vectors,
        )

    def get_index_health(self):
        """Return feature-version compatibility status for the indexed catalog."""
        return self._repo.get_feature_version_status()

    def execute(
        self,
        query_image_path: str,
        top_k: int = 20,
        filters: Optional[Dict[str, str]] = None,
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
            timer = PipelineTimer("SEARCH TIMING")

            with timer.measure("image_load"):
                if not validate_image(query_path):
                    raise ValueError(
                        f"Selected file is not a valid, readable image: {query_path.name}"
                    )
                query_sha256 = compute_sha256(query_path)
                query_dhash = compute_dhash(query_path)

            query_features: TileFeatures | None = None
            cached_tile = self._repo.get_by_path(str(query_path.resolve()))
            if (
                cached_tile
                and cached_tile.is_indexed
                and cached_tile.features is not None
                and cached_tile.sha256_hash == query_sha256
            ):
                query_features = cached_tile.features
                logger.info(
                    "Reusing indexed features for catalog query: %s",
                    query_path.name,
                )

            if query_features is None:
                logger.info("Computing embedding for query image...")
                with timer.measure("feature_extract"):
                    query_features = self._feature_extractor.extract(
                        str(query_path),
                        for_query=True,
                    )
                extract_timings = self._feature_extractor.last_timings
                timer.timings.record("preprocessing", extract_timings.preprocessing)
                timer.timings.record("dinov2", extract_timings.dinov2)
                timer.timings.record("descriptors", extract_timings.descriptors)
            else:
                timer.timings.record("preprocessing", 0.0)
                timer.timings.record("dinov2", 0.0)
                timer.timings.record("descriptors", 0.0)


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

            if total_vectors <= 0:
                logger.info("FAISS index is empty. No search results available.")
                return []

            filtered_ids: Optional[set[int]] = None
            if active_filters:
                filtered_ids = set(self._repo.get_ids_matching_filters(active_filters))
                if not filtered_ids:
                    logger.info(
                        "No indexed tiles match metadata filters: %s",
                        active_filters,
                    )
                    return []

            candidates: List[TileImage] = []

            if filtered_ids is not None and len(filtered_ids) <= _FILTERED_FULL_RERANK_CAP:
                logger.info(
                    "Metadata filters active — reranking %d filtered catalog tile(s) directly.",
                    len(filtered_ids),
                )
                with timer.measure("database"):
                    matched_tiles = self._repo.get_by_ids(list(filtered_ids))
                candidates = [
                    tile for tile in matched_tiles if tile.features is not None
                ]
            else:
                search_k = self._compute_faiss_search_k(top_k, total_vectors)

                if active_filters:
                    search_k = min(
                        max(top_k * _FILTER_CANDIDATE_MULTIPLIER, top_k),
                        _FILTER_CANDIDATE_CAP,
                        total_vectors,
                    )

                logger.info(f"Querying FAISS vector index (search_k={search_k})...")
                with timer.measure("faiss"):
                    matching_ids, _ = self._index.search_vectors(
                        query_features.embedding.tolist(),
                        search_k,
                    )

                if not matching_ids:
                    logger.info("No matching records found in vector index.")
                    return []

                logger.info(
                    "Retrieving database metadata for matching IDs: %s",
                    matching_ids,
                )
                with timer.measure("database"):
                    matched_tiles = self._repo.get_by_ids(matching_ids)

                tile_map = {t.id: t for t in matched_tiles if t.id is not None}

                for record_id in matching_ids:
                    tile = tile_map.get(record_id)
                    if tile is None:
                        continue
                    if filtered_ids is not None and record_id not in filtered_ids:
                        continue
                    if not self._matches_filters(tile, active_filters):
                        continue
                    candidates.append(tile)

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
                catalog_source_tile = self._find_catalog_tile_by_stem(crop_stem)
                if catalog_source_tile is not None:
                    logger.info(
                        "Crop search linked to catalog tile: %s",
                        catalog_source_tile.file_name,
                    )

            reranked = []

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

                    if (
                        not exact_match
                        and catalog_source_tile is not None
                        and tile.id == catalog_source_tile.id
                        and hybrid.embedding >= _CROP_SOURCE_EMBEDDING_THRESHOLD
                    ):
                        exact_match = True
                        final_score = 1.0
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

            reranked = self._filter_weak_results(reranked, top_k)

            results: List[SearchResult] = []

            for score, tile, exact_match in reranked[:top_k]:

                thumbnail_path = get_thumbnail_path(
                    Path(tile.file_path),
                    self._thumbnail_dir,
                )

                thumb_str = (
                    str(thumbnail_path)
                    if thumbnail_path.exists()
                    else tile.file_path
                )

                similarity_percentage = calibrate_display_percent(
                    score,
                    exact_match=exact_match,
                )

                results.append(
                    SearchResult(
                        tile=tile,
                        similarity_score=similarity_percentage,
                        thumbnail_path=thumb_str,
                    )
                )

            timer.log_summary(log=logger)
            return results
        except Exception as e:
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

        stem = query_path.stem
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

        logger.info(
            "Weak-result filter: kept %d of %d candidates (min_raw=%.3f)",
            len(kept),
            len(reranked),
            min_raw,
        )
        return kept
