"""
FAISS CPU vector index management module for TileVision AI.

Manages storing, updating, and querying high-dimensional vector embeddings
using an ID-mapped Flat Inner Product FAISS index.
"""

import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    # Fallback/mock support for environments where faiss is not pre-installed yet
    faiss = None

from src.ai.inference_guard import (
    DEFAULT_INDEX_LOCK_TIMEOUT_S,
    DEFAULT_SEARCH_LOCK_TIMEOUT_S,
    synchronized_inference,
)
from src.ai.index_metadata import read_index_metadata, write_index_metadata

logger = logging.getLogger("tilevision.ai.vector_index")


class FaissIndexManager:
    """
    Manages a local FAISS CPU vector index linked to SQLite database primary keys.
    dimension: Dimension of the vector embeddings (1024 for DINOv2 Large)
    """

    def __init__(self, index_path: str, dimension: int = 1024) -> None:
        """
        Initialize the index manager.

        Args:
            index_path: Absolute path to save/load the FAISS index.
            dimension: Dimension of the vector embeddings (e.g. 512 for CLIP ViT-B/32).
        """
        self._index_path = Path(index_path)
        self._dimension = dimension
        self._index = None

    @property
    def index_path(self) -> Path:
        """Absolute path to the FAISS index file on disk (Task A: Dashboard, FAISS Index Size)."""
        return self._index_path

    def load_index(self) -> None:
        """
        Load the index from disk.
        
        Creates a new IndexIDMap wrapping an IndexFlatIP (Inner Product)
        if the file does not exist.
        """
        
        if faiss is None:
            logger.critical("faiss package is not installed! Cannot load index.")
            raise ImportError("faiss-cpu package is required for FaissIndexManager.")

        # Parallelize FlatIP distance scans (exact O(n·d) search).
        # This is IndexFlatIP — NOT IVF/HNSW/PQ. OpenMP speeds linear scan;
        # asymptotic complexity remains O(n) in catalog size.
        try:
            import os

            threads = min(8, max(1, (os.cpu_count() or 4)))
            faiss.omp_set_num_threads(threads)
            logger.debug("FAISS omp threads set to %d", threads)
        except Exception as exc:
            logger.debug("Could not set FAISS omp threads: %s", exc)

        # Ensure parent folder exists
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

        with synchronized_inference(timeout=DEFAULT_INDEX_LOCK_TIMEOUT_S, purpose="FAISS"):
            if self._index_path.exists() and self._index_path.stat().st_size > 0:
                logger.info(f"Loading existing FAISS index from: {self._index_path}")
                meta = read_index_metadata(self._index_path)
                if meta is not None and not meta.is_compatible():
                    logger.warning(
                        "FAISS metadata incompatible "
                        "(model=%s dim=%s feature_v=%s app=%s). "
                        "Index will load but Settings → Rebuild FAISS Index is required "
                        "for correct results.",
                        meta.embedding_model,
                        meta.embedding_dimension,
                        meta.feature_version,
                        meta.app_version,
                    )
                try:
                    self._index = faiss.read_index(str(self._index_path))
                    logger.info(f"FAISS index loaded. Total vectors: {self._index.ntotal}")
                except Exception as e:
                    logger.error(f"Failed to load FAISS index from file: {e}. Creating new index.")
                    self._create_new_index()
            else:
                logger.info("No index file found. Initializing a new FAISS index.")
                self._create_new_index()

    def _create_new_index(self) -> None:
        """
        Initialize a new IndexIDMap wrapping IndexFlatIP.

        FlatIP performs exact inner-product (cosine on L2-normalized vectors)
        search in O(n·d). Approximate indexes (IVF/HNSW/PQ) are not used so
        showroom recall stays exact and rebuilds stay simple.
        """
        # Flat Inner Product index — exact search, O(n·d)
        flat_index = faiss.IndexFlatIP(self._dimension)
        # IDMap allows mapping SQLite database IDs (non-consecutive) to vectors
        self._index = faiss.IndexIDMap(flat_index)
        logger.info(
            "New empty FAISS IndexIDMap(IndexFlatIP) initialized "
            "(dimension=%d, complexity=O(n·d) exact).",
            self._dimension,
        )

    def index_type_name(self) -> str:
        """Human-readable FAISS index type for profiling (e.g. IndexFlatIP)."""
        if self._index is None:
            try:
                self.load_index()
            except Exception:
                return "unloaded"
        idx = self._index
        try:
            if faiss is not None and hasattr(idx, "index"):
                inner = faiss.downcast_index(idx.index)
                return f"IndexIDMap({type(inner).__name__})"
            if faiss is not None:
                return type(faiss.downcast_index(idx)).__name__
            return type(idx).__name__
        except Exception:
            return type(idx).__name__ if idx is not None else "unloaded"

    def embedding_dimension(self) -> int:
        return int(self._dimension)

    def add_vectors(
        self,
        ids: List[int],
        vectors: List[List[float]] | List[np.ndarray] | np.ndarray,
        persist: bool = True,
    ) -> None:
        """
        Add normalized vectors to the index, mapped to database record IDs.

        Args:
            ids: List of database primary key IDs.
            vectors: Embedding vectors as list-of-lists, list of ndarrays,
                or a 2D float32 array (preferred — no copy when contiguous).
            persist: If True (default), immediately writes the index to disk.
                Callers doing many small additions in a loop (e.g. a folder
                scan indexing hundreds/thousands of files) should pass False
                and call save_index() themselves periodically/at the end —
                writing the whole index to disk after every single file is
                extremely slow on large catalogs.
        """
        if self._index is None:
            self.load_index()

        if not ids:
            logger.warning("Empty ids provided to add_vectors. Skipping.")
            return

        try:
            with synchronized_inference(timeout=DEFAULT_INDEX_LOCK_TIMEOUT_S, purpose="FAISS add"):
                ids_np = np.asarray(ids, dtype=np.int64)
                vectors_np = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
                if vectors_np.ndim != 2:
                    raise ValueError("vectors must be a 2D array of shape (n, dim)")
                if vectors_np.shape[0] != len(ids):
                    raise ValueError("Size mismatch: The number of IDs must match the number of vectors.")

                if vectors_np.shape[1] != self._dimension:
                    raise ValueError(
                        f"Vector dimension mismatch. Index dimension: {self._dimension}, "
                        f"Provided vector dimension: {vectors_np.shape[1]}"
                    )

                self._index.add_with_ids(vectors_np, ids_np)
                logger.info(f"Added {len(ids)} vectors to FAISS index. Total now: {self._index.ntotal}")
            if persist:
                self.save_index()
        except Exception as e:
            logger.error(f"Failed to add vectors to FAISS index: {e}")
            raise RuntimeError(f"FAISS index write error: {e}") from e

    def update_vectors(
        self,
        ids: List[int],
        vectors: List[List[float]] | List[np.ndarray] | np.ndarray,
        persist: bool = True,
    ) -> None:
        """
        Replace vectors for ids that may already exist in the index.

        FAISS's IndexIDMap does NOT enforce unique ids: calling add_with_ids()
        with an id that is already present simply appends a second vector
        under that same id, rather than replacing it. Left unchecked, this
        means re-indexing a tile whose file content changed leaves the index
        with two entries for that tile (the stale embedding and the fresh
        one) — search results start returning the same tile twice, and
        remove_vectors() for that id removes both at once. This method
        removes any existing vector(s) for the given ids first, so the
        index always holds exactly one vector per id.

        Safe to call for brand-new ids too (remove_ids is a no-op if the id
        isn't present yet).

        Args:
            ids: List of database primary key IDs.
            vectors: List of embedding vectors (list of floats).
            persist: If True (default), writes the index to disk immediately.
        """
        if self._index is None:
            self.load_index()

        if not ids:
            return

        try:
            with synchronized_inference(timeout=DEFAULT_INDEX_LOCK_TIMEOUT_S, purpose="FAISS"):
                self._index.remove_ids(np.array(ids, dtype=np.int64))
        except Exception as e:
            logger.debug(f"No pre-existing vector(s) to remove for ids {ids} (or removal failed): {e}")

        self.add_vectors(ids, vectors, persist=persist)

    def remove_vectors(self, ids: List[int]) -> bool:
        """
        Remove vectors from the index by their database record IDs.

        Args:
            ids: List of database primary keys to remove.

        Returns:
            True if deletion completed, False otherwise.
        """
        if self._index is None:
            self.load_index()

        if not ids:
            return False

        try:
            removed_count = 0
            with synchronized_inference(timeout=DEFAULT_INDEX_LOCK_TIMEOUT_S, purpose="FAISS"):
                ids_np = np.array(ids, dtype=np.int64)
                # remove_ids returns number of removed elements
                removed_count = self._index.remove_ids(ids_np)
                logger.info(f"Removed {removed_count} vectors from FAISS index. Total remaining: {self._index.ntotal}")
            if removed_count > 0:
                self.save_index()
                return True
        except Exception as e:
            logger.error(f"Failed to remove IDs {ids} from FAISS index: {e}")
        return False

    def get_total_count(self) -> int:
        """
        Get the total number of vectors currently stored in the index.

        Returns:
            The vector count (0 if the index hasn't been loaded/created yet).
        """
        if self._index is None:
            self.load_index()
        # Brief lock so concurrent add/remove cannot tear the read.
        with synchronized_inference(
            timeout=DEFAULT_SEARCH_LOCK_TIMEOUT_S, purpose="FAISS ntotal"
        ):
            return int(self._index.ntotal) if self._index is not None else 0

    def search_vectors(
        self,
        query_vector: List[float] | np.ndarray,
        top_k: int,
    ) -> Tuple[List[int], List[float]]:
        """
        Search for the top_k closest vectors.

        Args:
            query_vector: Normalized 1D float list or ndarray (preferred).
            top_k: Number of results to retrieve.

        Returns:
            A tuple of (list of integer database IDs, list of similarity scores).
        """
        if self._index is None:
            self.load_index()

        if self._index.ntotal == 0:
            logger.info("FAISS index is empty. Returning empty search results.")
            return [], []

        try:
            with synchronized_inference(
                timeout=DEFAULT_SEARCH_LOCK_TIMEOUT_S, purpose="FAISS search"
            ):
                # Prefer ndarray input — avoids list→array roundtrip on hot path.
                query_np = np.ascontiguousarray(
                    np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
                )
                if query_np.shape[1] != self._index.d:
                    raise ValueError(
                        f"Query dimension {query_np.shape[1]} != index dimension {self._index.d}"
                    )

                norm = np.linalg.norm(query_np, axis=1, keepdims=True)
                norm = np.maximum(norm, 1e-12)
                query_np = query_np / norm

                safe_top_k = min(max(int(top_k), 1), int(self._index.ntotal))

                logger.debug(
                    "FAISS search: dimension=%d query_shape=%s query_norm=%.4f",
                    self._index.d,
                    query_np.shape,
                    float(np.linalg.norm(query_np)),
                )
                scores, indices = self._index.search(query_np, safe_top_k)

                indices_flat = indices[0]
                scores_flat = scores[0]

                matching_ids: List[int] = []
                similarity_scores: List[float] = []

                for idx, score in zip(indices_flat.tolist(), scores_flat.tolist()):
                    if idx != -1:
                        matching_ids.append(int(idx))
                        clamped_score = max(-1.0, min(1.0, float(score)))
                        similarity_scores.append(clamped_score)

                return matching_ids, similarity_scores
        except Exception:
            logger.exception("FAISS vector search failed")
            raise

    def save_index(self) -> None:
        """Write the current state of the FAISS index to disk."""
        if self._index is None:
            return

        try:
            with synchronized_inference(timeout=DEFAULT_INDEX_LOCK_TIMEOUT_S, purpose="FAISS"):
                # Ensure folder exists
                self._index_path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self._index, str(self._index_path))
                logger.info(f"FAISS index successfully saved to: {self._index_path}")
            try:
                write_index_metadata(
                    self._index_path,
                    faiss_type=self.index_type_name(),
                    ntotal=int(self._index.ntotal),
                )
            except Exception as meta_exc:
                logger.warning("FAISS metadata sidecar write failed: %s", meta_exc)
        except Exception as e:
            logger.error(f"Failed to write FAISS index to {self._index_path}: {e}")
            raise OSError(f"FAISS index save failure: {e}") from e

    def clear_all(self) -> None:
        """Reset the index and delete the binary file."""
        with synchronized_inference(timeout=DEFAULT_INDEX_LOCK_TIMEOUT_S, purpose="FAISS"):
            self._create_new_index()
            try:
                if self._index_path.exists():
                    self._index_path.unlink()
                logger.info("Cleared FAISS index database file from disk.")
            except Exception as e:
                logger.error(f"Failed to delete FAISS index binary: {e}")
