"""
Optional FAISS index backends for TileVision AI Enterprise (v1.2+).

Production default remains IndexFlatIP (exact inner-product / cosine on
L2-normalized vectors). Approximate backends are opt-in via settings and
require a full rebuild when switching.

Backends
--------
flat_ip  — IndexIDMap(IndexFlatIP)     exact, O(n·d), production default
hnsw     — IndexIDMap2(IndexHNSWFlat)  approximate graph, fast at 1M+
ivf      — IndexIDMap2(IndexIVFFlat)   approximate inverted file
ivf_pq   — IndexIDMap2(IndexIVFPQ)     approximate + product quantization (RAM)

Accuracy note: only flat_ip preserves exact Top-K identity with the current
production ranking path. Approximate backends trade recall for latency/RAM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger("tilevision.ai.index_backends")

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


class IndexBackend(str, Enum):
    FLAT_IP = "flat_ip"
    HNSW = "hnsw"
    IVF = "ivf"
    IVF_PQ = "ivf_pq"

    @classmethod
    def parse(cls, value: str | None) -> "IndexBackend":
        raw = str(value or cls.FLAT_IP.value).strip().lower().replace("-", "_")
        aliases = {
            "flat": cls.FLAT_IP,
            "flatip": cls.FLAT_IP,
            "indexflatip": cls.FLAT_IP,
            "hnsw_flat": cls.HNSW,
            "ivfflat": cls.IVF,
            "ivfpq": cls.IVF_PQ,
        }
        if raw in aliases:
            return aliases[raw]
        try:
            return cls(raw)
        except ValueError:
            logger.warning("Unknown index_backend=%r; falling back to flat_ip", value)
            return cls.FLAT_IP


@dataclass(frozen=True, slots=True)
class BackendParams:
    """Tunables for optional approximate indexes."""

    hnsw_m: int = 32
    hnsw_ef_construction: int = 40
    hnsw_ef_search: int = 64
    ivf_nlist: int = 0  # 0 → auto from catalog size at train time
    ivf_nprobe: int = 16
    ivf_pq_m: int = 16  # must divide dimension
    ivf_pq_nbits: int = 8


@dataclass(frozen=True, slots=True)
class BackendMemoryEstimate:
    backend: IndexBackend
    vectors_mib: float
    overhead_mib: float
    total_mib: float
    notes: str


def estimate_index_memory_mib(
    *,
    ntotal: int,
    dimension: int,
    backend: IndexBackend,
    params: BackendParams | None = None,
) -> BackendMemoryEstimate:
    """Rough peak RAM for the FAISS structure alone (excludes DINOv2 / UI)."""
    params = params or BackendParams()
    n = max(0, int(ntotal))
    d = max(1, int(dimension))
    vectors = (n * d * 4) / (1024.0 * 1024.0)
    if backend is IndexBackend.FLAT_IP:
        overhead = vectors * 0.05
        notes = "Exact; RAM ≈ 4·n·d bytes"
    elif backend is IndexBackend.HNSW:
        # Rough graph storage: ~M links × 8 bytes × 2 (bi-directional) per vector
        overhead = (n * params.hnsw_m * 16) / (1024.0 * 1024.0)
        notes = "Exact vectors + HNSW graph links"
    elif backend is IndexBackend.IVF:
        overhead = vectors * 0.08
        notes = "Exact vectors + inverted lists"
    else:  # IVF_PQ
        # PQ codes: n × m bytes (approx) instead of full float vectors
        codes = (n * params.ivf_pq_m) / (1024.0 * 1024.0)
        codebook = (params.ivf_pq_m * (2**params.ivf_pq_nbits) * (d // max(params.ivf_pq_m, 1)) * 4) / (
            1024.0 * 1024.0
        )
        vectors = codes
        overhead = codebook + max(1.0, n * 0.0001)
        notes = "Compressed PQ codes (lossy); lowest RAM at 1M+"
    return BackendMemoryEstimate(
        backend=backend,
        vectors_mib=round(vectors, 1),
        overhead_mib=round(overhead, 1),
        total_mib=round(vectors + overhead, 1),
        notes=notes,
    )


def auto_nlist(ntotal: int, configured: int = 0) -> int:
    """Choose IVF nlist. Rule of thumb: ~4√n, clamped."""
    if configured > 0:
        return max(1, int(configured))
    n = max(1, int(ntotal))
    guess = int(4 * (n**0.5))
    return max(1, min(16384, guess, n))


def create_empty_index(
    *,
    dimension: int,
    backend: IndexBackend,
    params: BackendParams | None = None,
) -> Any:
    """
    Create an empty FAISS index for the requested backend.

    IVF / IVF-PQ indexes are returned untrained; callers must train before add.
    FlatIP keeps IndexIDMap (not IDMap2) for binary continuity with v1.0/v1.1.
    """
    if faiss is None:
        raise ImportError("faiss-cpu package is required")
    params = params or BackendParams()
    d = int(dimension)

    if backend is IndexBackend.FLAT_IP:
        return faiss.IndexIDMap(faiss.IndexFlatIP(d))

    if backend is IndexBackend.HNSW:
        inner = faiss.IndexHNSWFlat(d, int(params.hnsw_m), faiss.METRIC_INNER_PRODUCT)
        inner.hnsw.efConstruction = int(params.hnsw_ef_construction)
        inner.hnsw.efSearch = int(params.hnsw_ef_search)
        return faiss.IndexIDMap2(inner)

    if backend is IndexBackend.IVF:
        quantizer = faiss.IndexFlatIP(d)
        nlist = auto_nlist(max(params.ivf_nlist, 1) if params.ivf_nlist else 256, params.ivf_nlist)
        # Start with a modest nlist; may be rebuilt at train time if needed.
        inner = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        inner.nprobe = int(params.ivf_nprobe)
        return faiss.IndexIDMap2(inner)

    # IVF-PQ
    quantizer = faiss.IndexFlatIP(d)
    nlist = auto_nlist(max(params.ivf_nlist, 1) if params.ivf_nlist else 256, params.ivf_nlist)
    m = int(params.ivf_pq_m)
    if d % m != 0:
        # Fall back to a divisor of d closest to requested m.
        for cand in (16, 8, 4, 2, 1):
            if d % cand == 0:
                m = cand
                break
    inner = faiss.IndexIVFPQ(
        quantizer, d, nlist, m, int(params.ivf_pq_nbits), faiss.METRIC_INNER_PRODUCT
    )
    inner.nprobe = int(params.ivf_nprobe)
    return faiss.IndexIDMap2(inner)


def unwrap_inner(index: Any) -> Any:
    """Return the innermost searchable FAISS index (past IDMap wrappers)."""
    if faiss is None or index is None:
        return index
    try:
        cur = index
        for _ in range(4):
            if hasattr(cur, "index"):
                cur = faiss.downcast_index(cur.index)
            else:
                break
        return faiss.downcast_index(cur)
    except Exception:
        return index


def detect_backend(index: Any) -> IndexBackend:
    """Best-effort backend detection from a loaded FAISS object."""
    inner = unwrap_inner(index)
    name = type(inner).__name__.lower()
    if "ivfpq" in name:
        return IndexBackend.IVF_PQ
    if "ivf" in name:
        return IndexBackend.IVF
    if "hnsw" in name:
        return IndexBackend.HNSW
    return IndexBackend.FLAT_IP


def min_ivf_train_points(backend: IndexBackend, params: BackendParams, nlist: int) -> int:
    """Minimum vectors required before IVF / IVF-PQ training is safe."""
    if backend is IndexBackend.IVF:
        return max(int(nlist), 1)
    if backend is IndexBackend.IVF_PQ:
        return max(int(nlist), int(2**params.ivf_pq_nbits), 39)
    return 0


def ensure_trained(index: Any, vectors_np, *, params: BackendParams | None = None) -> Any:
    """
    Train IVF-family indexes if needed (no-op for FlatIP / HNSW).

    May replace the inner IVF structure so nlist / PQ nbits fit the training
    set. Returns the (possibly replaced) index object.
    """
    if faiss is None:
        return index
    params = params or BackendParams()
    inner = unwrap_inner(index)
    if getattr(inner, "is_trained", True):
        return index

    n = int(vectors_np.shape[0])
    d = int(vectors_np.shape[1])
    backend = detect_backend(index)
    desired = min(auto_nlist(n, params.ivf_nlist), max(1, n))
    rebuild = False
    nbits = int(params.ivf_pq_nbits)

    if hasattr(inner, "nlist") and int(inner.nlist) != desired:
        rebuild = True
    if backend is IndexBackend.IVF_PQ:
        # Hard floor avoids FAISS PQ segfaults; soft floor is ~39× codewords.
        hard = max(desired, int(2**params.ivf_pq_nbits))
        soft = max(desired * 39, int(2**params.ivf_pq_nbits) * 39)
        if n < hard:
            logger.warning(
                "IVF-PQ needs at least %d training vectors (got %d); using IVF-Flat.",
                hard,
                n,
            )
            backend = IndexBackend.IVF
            rebuild = True
        elif n < soft:
            logger.warning(
                "IVF-PQ training set small (%d < recommended %d); quality may be low.",
                n,
                soft,
            )
            nbits = int(params.ivf_pq_nbits)
        else:
            nbits = int(params.ivf_pq_nbits)

    if rebuild or (hasattr(inner, "nlist") and int(inner.nlist) > n):
        logger.info(
            "Building untrained %s with nlist=%d for training size=%d",
            backend.value,
            desired,
            n,
        )
        quantizer = faiss.IndexFlatIP(d)
        if backend is IndexBackend.IVF_PQ:
            m = int(params.ivf_pq_m)
            if d % m != 0:
                m = next(c for c in (16, 8, 4, 2, 1) if d % c == 0)
            new_inner = faiss.IndexIVFPQ(
                quantizer, d, desired, m, nbits, faiss.METRIC_INNER_PRODUCT
            )
        else:
            new_inner = faiss.IndexIVFFlat(
                quantizer, d, desired, faiss.METRIC_INNER_PRODUCT
            )
        new_inner.nprobe = min(int(params.ivf_nprobe), desired)
        if hasattr(index, "index"):
            index.index = new_inner
        else:
            index = faiss.IndexIDMap2(new_inner)
        inner = new_inner

    logger.info("Training FAISS %s on %d vectors...", type(inner).__name__, n)
    inner.train(vectors_np)
    return index


def apply_search_params(index: Any, params: BackendParams) -> None:
    """Apply runtime search tunables (efSearch / nprobe)."""
    if faiss is None or index is None:
        return
    inner = unwrap_inner(index)
    try:
        if hasattr(inner, "hnsw"):
            inner.hnsw.efSearch = int(params.hnsw_ef_search)
        if hasattr(inner, "nprobe"):
            inner.nprobe = int(params.ivf_nprobe)
    except Exception as exc:
        logger.debug("Could not apply search params: %s", exc)


def backend_display_name(backend: IndexBackend) -> str:
    return {
        IndexBackend.FLAT_IP: "IndexFlatIP (exact, production default)",
        IndexBackend.HNSW: "HNSW (optional approximate)",
        IndexBackend.IVF: "IVF-Flat (optional approximate)",
        IndexBackend.IVF_PQ: "IVF-PQ (optional approximate + compressed)",
    }[backend]
