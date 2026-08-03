"""
FAISS CPU vector index management module for TileVision AI.

Manages storing, updating, and querying high-dimensional vector embeddings.
Production default: IndexIDMap(IndexFlatIP) — exact inner product.

Enterprise (v1.2+): optional HNSW / IVF / IVF-PQ backends via settings.
Switching backends requires a guided rebuild; FlatIP remains the default.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from src.ai.index_backends import (
    BackendParams,
    IndexBackend,
    apply_search_params,
    auto_nlist,
    create_empty_index,
    detect_backend,
    ensure_trained,
    min_ivf_train_points,
    unwrap_inner,
)
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

    def __init__(
        self,
        index_path: str,
        dimension: int = 1024,
        *,
        backend: IndexBackend | str = IndexBackend.FLAT_IP,
        backend_params: BackendParams | None = None,
    ) -> None:
        self._index_path = Path(index_path)
        self._dimension = dimension
        self._index = None
        self._backend = IndexBackend.parse(
            backend.value if isinstance(backend, IndexBackend) else backend
        )
        self._params = backend_params or BackendParams()
        self._pending_ids: list[int] = []
        self._pending_vectors: list[np.ndarray] = []

    @property
    def index_path(self) -> Path:
        """Absolute path to the FAISS index file on disk."""
        return self._index_path

    @property
    def configured_backend(self) -> IndexBackend:
        """Backend requested by settings (production: flat_ip)."""
        return self._backend

    def configure_backend(self, backend: IndexBackend | str) -> None:
        """Update configured backend (disk rewrite happens on clear/rebuild)."""
        self._backend = IndexBackend.parse(
            backend.value if isinstance(backend, IndexBackend) else backend
        )

    def active_backend(self) -> IndexBackend:
        """Backend of the currently loaded FAISS object."""
        if self._index is None:
            return self._backend
        return detect_backend(self._index)

    def load_index(self) -> None:
        """
        Load the index from disk.

        Creates a new empty index for the configured backend if the file
        does not exist. Existing files are loaded as-is (backend detected).
        """
        if faiss is None:
            logger.critical("faiss package is not installed! Cannot load index.")
            raise ImportError("faiss-cpu package is required for FaissIndexManager.")

        try:
            import os

            from src.utils.platform_info import is_mac_intel

            # Match torch: single OpenMP thread on macOS Intel to avoid hangs.
            if is_mac_intel():
                threads = 1
            else:
                threads = min(8, max(1, (os.cpu_count() or 4)))
            faiss.omp_set_num_threads(threads)
            logger.debug("FAISS omp threads set to %d", threads)
        except Exception as exc:
            logger.debug("Could not set FAISS omp threads: %s", exc)

        self._index_path.parent.mkdir(parents=True, exist_ok=True)

        with synchronized_inference(timeout=DEFAULT_INDEX_LOCK_TIMEOUT_S, purpose="FAISS"):
            if self._index_path.exists() and self._index_path.stat().st_size > 0:
                logger.info("Loading existing FAISS index from: %s", self._index_path)
                meta = read_index_metadata(self._index_path)
                if meta is not None and not meta.is_compatible():
                    logger.warning(
                        "FAISS metadata incompatible "
                        "(model=%s dim=%s feature_v=%s app=%s backend=%s). "
                        "Index will load but Settings → Rebuild FAISS Index is required "
                        "for correct results.",
                        meta.embedding_model,
                        meta.embedding_dimension,
                        meta.feature_version,
                        meta.app_version,
                        getattr(meta, "index_backend", "?"),
                    )
                try:
                    self._index = faiss.read_index(str(self._index_path))
                    apply_search_params(self._index, self._params)
                    loaded = detect_backend(self._index)
                    if loaded != self._backend:
                        logger.warning(
                            "Loaded FAISS backend=%s differs from configured=%s. "
                            "Search uses the on-disk index until a rebuild.",
                            loaded.value,
                            self._backend.value,
                        )
                    logger.info(
                        "FAISS index loaded. backend=%s vectors=%d",
                        loaded.value,
                        self._index.ntotal,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to load FAISS index from file: %s. Creating new index.", e
                    )
                    self._create_new_index()
            else:
                logger.info(
                    "No index file found. Initializing new FAISS index (backend=%s).",
                    self._backend.value,
                )
                self._create_new_index()

    def _create_new_index(self) -> None:
        """Initialize an empty index for the configured backend."""
        self._index = create_empty_index(
            dimension=self._dimension,
            backend=self._backend,
            params=self._params,
        )
        apply_search_params(self._index, self._params)
        logger.info(
            "New empty FAISS index initialized (backend=%s, dimension=%d).",
            self._backend.value,
            self._dimension,
        )

    def index_type_name(self) -> str:
        """Human-readable FAISS index type for profiling / diagnostics."""
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

        IVF / IVF-PQ indexes buffer until enough vectors exist to train safely,
        then train + flush in one shot (important for small indexing batches).
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
                    raise ValueError(
                        "Size mismatch: The number of IDs must match the number of vectors."
                    )
                if vectors_np.shape[1] != self._dimension:
                    raise ValueError(
                        f"Vector dimension mismatch. Index dimension: {self._dimension}, "
                        f"Provided vector dimension: {vectors_np.shape[1]}"
                    )

                inner = unwrap_inner(self._index)
                needs_train = not bool(getattr(inner, "is_trained", True))
                if needs_train and self._backend in (IndexBackend.IVF, IndexBackend.IVF_PQ):
                    nlist_hint = int(getattr(inner, "nlist", 0) or 0)
                    if nlist_hint <= 0:
                        nlist_hint = auto_nlist(max(len(ids), 256), self._params.ivf_nlist)
                    required = min_ivf_train_points(self._backend, self._params, nlist_hint)
                    for i, vec in enumerate(vectors_np):
                        self._pending_ids.append(int(ids_np[i]))
                        self._pending_vectors.append(np.asarray(vec, dtype=np.float32).copy())
                    if len(self._pending_ids) < required and not persist:
                        logger.info(
                            "Buffered %d/%d vectors for %s training",
                            len(self._pending_ids),
                            required,
                            self._backend.value,
                        )
                        return
                    # Enough vectors, or caller asked to persist → train with what we have
                    # (ensure_trained adapts nlist / PQ nbits / IVF-Flat fallback).
                    ids_np = np.asarray(self._pending_ids, dtype=np.int64)
                    vectors_np = np.ascontiguousarray(
                        np.stack(self._pending_vectors).astype(np.float32)
                    )
                    self._pending_ids.clear()
                    self._pending_vectors.clear()

                self._index = ensure_trained(self._index, vectors_np, params=self._params)
                apply_search_params(self._index, self._params)
                self._index.add_with_ids(vectors_np, ids_np)
                logger.info(
                    "Added %d vectors to FAISS index. Total now: %d",
                    int(vectors_np.shape[0]),
                    self._index.ntotal,
                )
            if persist:
                self.save_index()
        except Exception as e:
            logger.error("Failed to add vectors to FAISS index: %s", e)
            raise RuntimeError(f"FAISS index write error: {e}") from e

    def flush_pending_train(self, persist: bool = True) -> None:
        """Force-train IVF backends with whatever is buffered (adapts nlist/nbits)."""
        if not self._pending_ids:
            return
        ids = list(self._pending_ids)
        vectors = np.stack(self._pending_vectors).astype(np.float32)
        self._pending_ids.clear()
        self._pending_vectors.clear()
        # persist=True forces train even if below ideal sample count.
        self.add_vectors(ids, vectors, persist=persist)

    def update_vectors(
        self,
        ids: List[int],
        vectors: List[List[float]] | List[np.ndarray] | np.ndarray,
        persist: bool = True,
    ) -> None:
        """
        Replace vectors for ids that may already exist in the index.

        Removes any existing vector(s) for the given ids first so the index
        always holds exactly one vector per id.
        """
        if self._index is None:
            self.load_index()

        if not ids:
            return

        try:
            with synchronized_inference(timeout=DEFAULT_INDEX_LOCK_TIMEOUT_S, purpose="FAISS"):
                self._index.remove_ids(np.array(ids, dtype=np.int64))
        except Exception as e:
            logger.debug(
                "No pre-existing vector(s) to remove for ids %s (or removal failed): %s",
                ids,
                e,
            )

        self.add_vectors(ids, vectors, persist=persist)

    def remove_vectors(self, ids: List[int]) -> bool:
        """Remove vectors from the index by their database record IDs."""
        if self._index is None:
            self.load_index()

        if not ids:
            return False

        try:
            removed_count = 0
            with synchronized_inference(timeout=DEFAULT_INDEX_LOCK_TIMEOUT_S, purpose="FAISS"):
                ids_np = np.array(ids, dtype=np.int64)
                removed_count = self._index.remove_ids(ids_np)
                logger.info(
                    "Removed %d vectors from FAISS index. Total remaining: %d",
                    removed_count,
                    self._index.ntotal,
                )
            if removed_count > 0:
                self.save_index()
                return True
        except Exception as e:
            logger.error("Failed to remove IDs %s from FAISS index: %s", ids, e)
        return False

    def get_total_count(self) -> int:
        """Get the total number of vectors currently stored in the index."""
        if self._index is None:
            self.load_index()
        with synchronized_inference(
            timeout=DEFAULT_SEARCH_LOCK_TIMEOUT_S, purpose="FAISS ntotal"
        ):
            return int(self._index.ntotal) if self._index is not None else 0

    def search_vectors(
        self,
        query_vector: List[float] | np.ndarray,
        top_k: int,
    ) -> Tuple[List[int], List[float]]:
        """Search for the top_k closest vectors (exact for FlatIP)."""
        if self._index is None:
            self.load_index()

        if self._index.ntotal == 0:
            logger.info("FAISS index is empty. Returning empty search results.")
            return [], []

        try:
            with synchronized_inference(
                timeout=DEFAULT_SEARCH_LOCK_TIMEOUT_S, purpose="FAISS search"
            ):
                apply_search_params(self._index, self._params)
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
                    "FAISS search: backend=%s dimension=%d query_shape=%s",
                    self.active_backend().value,
                    self._index.d,
                    query_np.shape,
                )
                scores, indices = self._index.search(query_np, safe_top_k)

                matching_ids: List[int] = []
                similarity_scores: List[float] = []
                for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
                    if idx != -1:
                        matching_ids.append(int(idx))
                        similarity_scores.append(max(-1.0, min(1.0, float(score))))

                return matching_ids, similarity_scores
        except Exception:
            logger.exception("FAISS vector search failed")
            raise

    def save_index(self) -> None:
        """Write the current state of the FAISS index to disk."""
        if self._pending_ids:
            # Finish IVF training before persisting so catalogs aren't empty on disk.
            self.flush_pending_train(persist=False)

        if self._index is None:
            return

        try:
            with synchronized_inference(timeout=DEFAULT_INDEX_LOCK_TIMEOUT_S, purpose="FAISS"):
                self._index_path.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self._index, str(self._index_path))
                logger.info("FAISS index successfully saved to: %s", self._index_path)
            try:
                write_index_metadata(
                    self._index_path,
                    faiss_type=self.index_type_name(),
                    ntotal=int(self._index.ntotal),
                    index_backend=self.active_backend().value,
                )
            except Exception as meta_exc:
                logger.warning("FAISS metadata sidecar write failed: %s", meta_exc)
        except Exception as e:
            logger.error("Failed to write FAISS index to %s: %s", self._index_path, e)
            raise OSError(f"FAISS index save failure: {e}") from e

    def clear_all(self) -> None:
        """Reset the index and delete the binary file."""
        with synchronized_inference(timeout=DEFAULT_INDEX_LOCK_TIMEOUT_S, purpose="FAISS"):
            self._create_new_index()
            try:
                if self._index_path.exists():
                    self._index_path.unlink()
                meta = self._index_path.with_suffix(self._index_path.suffix + ".meta.json")
                if meta.exists():
                    meta.unlink()
                logger.info("Cleared FAISS index database file from disk.")
            except Exception as e:
                logger.error("Failed to delete FAISS index binary: %s", e)
