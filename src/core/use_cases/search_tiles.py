"""
Visual similarity search use case module for TileVision AI.

Given a query image, extracts features, performs FAISS vector search, and merges
matching items with SQLite database metadata and cached thumbnail paths.
"""

import logging
from pathlib import Path
from typing import List

from src.core.models import TileImage, SearchResult
from src.data.repository_interface import IImageRepository
from src.ai.embedder import OpenCLIPEmbedder
from src.ai.vector_index import FaissIndexManager
from src.utils.image_utils import get_thumbnail_path, validate_image

logger = logging.getLogger("tilevision.core.use_cases.search_tiles")


class SearchTilesUseCase:
    """
    Use case to query visual similarity of a tile sample against the indexed catalog.
    """

    def __init__(
        self,
        image_repository: IImageRepository,
        embedder: OpenCLIPEmbedder,
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
        self._embedder = embedder
        self._index = vector_index
        self._thumbnail_dir = Path(thumbnail_dir)

    def execute(self, query_image_path: str, top_k: int = 20) -> List[SearchResult]:
        """
        Execute visual similarity search for a query tile image.

        Args:
            query_image_path: Absolute path to the user's target search image.
            top_k: Maximum number of closest matches to return.

        Returns:
            A list of SearchResult objects sorted by similarity score descending.
        """
        query_path = Path(query_image_path)
        if not query_path.exists() or not query_path.is_file():
            raise FileNotFoundError(f"Query image does not exist: {query_image_path}")

        if not validate_image(query_path):
            raise ValueError(f"Selected file is not a valid, readable image: {query_path.name}")

        top_k = max(1, int(top_k))

        logger.info(f"Initiating similarity search query for: {query_path.name} (top_k={top_k})")
        
        try:
            # 1. Generate query embedding from CLIP model
            logger.info("Computing embedding for query image...")
            query_embedding = self._embedder.get_embedding(str(query_path))

            # 2. Search FAISS index for top K matching IDs
            logger.info("Querying FAISS vector index...")
            matching_ids, similarity_scores = self._index.search_vectors(query_embedding, top_k)
            
            if not matching_ids:
                logger.info("No matching records found in vector index.")
                return []

            # 3. Retrieve metadata records from SQLite in the exact order of matches
            logger.info(f"Retrieving database metadata for matching IDs: {matching_ids}")
            matched_tiles = self._repo.get_by_ids(matching_ids)
            
            # Create a lookup map of Tile ID -> TileImage
            tile_map = {t.id: t for t in matched_tiles if t.id is not None}

            # 4. Construct SearchResult list mapping matches and cached thumbnail paths
            results: List[SearchResult] = []
            for record_id, score in zip(matching_ids, similarity_scores):
                if record_id in tile_map:
                    tile = tile_map[record_id]
                    
                    # Resolve cached thumbnail path (same hashing logic used at index time)
                    thumbnail_path = get_thumbnail_path(Path(tile.file_path), self._thumbnail_dir)

                    # Fallback to the full-size source image if no cached thumbnail exists
                    # (e.g. thumbnail generation previously failed for this file)
                    thumb_str = str(thumbnail_path) if thumbnail_path.exists() else tile.file_path

                    # Map score (cosine similarity range -1.0 to 1.0) to 0-100 percentage
                    # For CLIP cosine similarity, negative scores are extremely rare.
                    similarity_percentage = max(0.0, min(100.0, score * 100.0))

                    results.append(
                        SearchResult(
                            tile=tile,
                            similarity_score=similarity_percentage,
                            thumbnail_path=thumb_str,
                        )
                    )

            logger.info(f"Search query completed. Found {len(results)} matches.")
            return results
        except Exception as e:
            logger.error(f"Failed to execute tile search query: {e}")
            raise RuntimeError(f"Visual similarity search execution error: {e}") from e
