"""
Visual similarity search use case module for TileVision AI.

Given a query image, extracts features, performs FAISS vector search, and merges
matching items with SQLite database metadata and cached thumbnail paths.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.ai.candidate_filter import CandidateFilter
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

# Perceptual hashes within this Hamming distance are treated as near-exact
# self matches (same tile, different compression/crop).
_NEAR_EXACT_DHASH_THRESHOLD = 3


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

        try:
            timer = PipelineTimer("SEARCH TIMING")

            with timer.measure("image_load"):
                if not validate_image(query_path):
                    raise ValueError(
                        f"Selected file is not a valid, readable image: {query_path.name}"
                    )
                query_sha256 = compute_sha256(query_path)
                query_dhash = compute_dhash(query_path)

            logger.info("Computing embedding for query image...")
            query_features = self._feature_extractor.extract(
                str(query_path)
            )
            extract_timings = self._feature_extractor.last_timings
            timer.timings.record("preprocessing", extract_timings.preprocessing)
            timer.timings.record("dinov2", extract_timings.dinov2)
            timer.timings.record("descriptors", extract_timings.descriptors)


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


            # 2. Search FAISS index. If filters are active, widen the
            #    candidate pool since some matches will be filtered out
            #    downstream — otherwise a filtered search could return
            #    fewer than top_k results even when more matches exist
            #    further down the similarity ranking.
            total_vectors = self._index.get_total_count()

            if total_vectors <= 0:
                logger.info("FAISS index is empty. No search results available.")
                return []

            search_k = min(
                max(top_k * 5, 100),
                total_vectors,
            )

            if active_filters:
                search_k = min(
                    max(
                        top_k * _FILTER_CANDIDATE_MULTIPLIER,
                        top_k,
                    ),
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

            # 3. Retrieve metadata records from SQLite in the exact order of matches
            logger.info(f"Retrieving database metadata for matching IDs: {matching_ids}")
            with timer.measure("database"):
                matched_tiles = self._repo.get_by_ids(matching_ids)
            
            # Create a lookup map of Tile ID -> TileImage
            tile_map = {t.id: t for t in matched_tiles if t.id is not None}

            # ----------------------------------------
            # Build candidate list
            # ----------------------------------------

            candidates = []

            for record_id in matching_ids:

                tile = tile_map.get(record_id)

                if tile is None:
                    continue

                if not self._matches_filters(tile, active_filters):
                    continue

                candidates.append(tile)

            logger.info(
                "Candidates before filtering: %d",
                len(candidates),
            )

            # ----------------------------------------
            # Candidate filtering
            # ----------------------------------------

            candidates = CandidateFilter.filter(
                query_features,
                candidates,
            )

            logger.info(
                "Candidates after filtering: %d",
                len(candidates),
            )
            
            # -------------------------------------------------------
            # Hybrid Re-ranking
            # -------------------------------------------------------

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
