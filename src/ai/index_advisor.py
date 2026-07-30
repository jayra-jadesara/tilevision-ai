"""
Index Advisor for TileVision AI.

Recommends an optimal FAISS backend from catalog size, RAM, CPU, and
accuracy targets. Never mutates settings or indexes — callers must ask
the user before applying a recommendation.

Production default remains IndexFlatIP (exact).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from src.ai.feature_versions import CURRENT_EMBEDDING_DIMENSION
from src.ai.index_backends import (
    BackendParams,
    IndexBackend,
    backend_display_name,
    estimate_index_memory_mib,
)


# Catalog size thresholds (inclusive upper bounds for bands).
_BAND_SMALL = 100_000
_BAND_MEDIUM = 500_000
_BAND_LARGE = 1_000_000

# Leave headroom for DINOv2 (~1–1.5 GiB), Qt, SQLite, OS.
_APP_HEADROOM_MIB = 3072.0
_INDEX_RAM_FRACTION = 0.40


@dataclass(frozen=True, slots=True)
class BackendComparisonRow:
    backend: str
    exact: bool
    recall: str
    memory: str
    speed: str
    best_use_case: str


@dataclass(frozen=True, slots=True)
class IndexAdvice:
    """User-facing recommendation snapshot (read-only)."""

    recommended_backend: IndexBackend
    reason: str
    estimated_ram_mib: float
    expected_recall: float
    expected_search_ms: float
    rebuild_required: bool
    optional_backend: Optional[IndexBackend]
    optional_note: str
    approximate_warning: str
    catalog_size: int
    embedding_dimension: int
    available_ram_mib: float | None
    cpu_cores: int
    current_backend: IndexBackend
    comparison: tuple[BackendComparisonRow, ...] = field(default_factory=tuple)

    @property
    def recommended_label(self) -> str:
        return backend_display_name(self.recommended_backend)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_backend": self.recommended_backend.value,
            "recommended_label": self.recommended_label,
            "reason": self.reason,
            "estimated_ram_mib": self.estimated_ram_mib,
            "expected_recall": self.expected_recall,
            "expected_search_ms": self.expected_search_ms,
            "rebuild_required": self.rebuild_required,
            "optional_backend": (
                self.optional_backend.value if self.optional_backend else None
            ),
            "optional_note": self.optional_note,
            "approximate_warning": self.approximate_warning,
            "catalog_size": self.catalog_size,
            "embedding_dimension": self.embedding_dimension,
            "available_ram_mib": self.available_ram_mib,
            "cpu_cores": self.cpu_cores,
            "current_backend": self.current_backend.value,
            "comparison": [asdict(row) for row in self.comparison],
        }


def _total_ram_mib() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        return None
    return None


def _usable_index_ram_mib(total_ram_mib: float | None) -> float:
    """Conservative budget for the FAISS structure alone."""
    if total_ram_mib is None or total_ram_mib <= 0:
        return 4096.0
    by_fraction = total_ram_mib * _INDEX_RAM_FRACTION
    by_headroom = max(512.0, total_ram_mib - _APP_HEADROOM_MIB)
    return max(512.0, min(by_fraction, by_headroom))


def estimate_search_latency_ms(
    backend: IndexBackend,
    *,
    catalog_size: int,
    dimension: int,
    cpu_cores: int,
) -> float:
    """
    Order-of-magnitude search latency for advisor UI (not a microbenchmark).

    Calibrated from synthetic FlatIP ~0.1 ms @ 10k×64-d on a few cores,
    scaled linearly in n·d / cores for FlatIP and discounted for ANN.
    """
    n = max(1, int(catalog_size))
    d = max(1, int(dimension))
    cores = max(1, int(cpu_cores))
    # Empirical scale: 10k * 64 / 4 ≈ 1.6e5 work units → ~0.12 ms
    flat_ms = (n * d) / (cores * 5.0e7) * 1000.0
    flat_ms = max(0.05, flat_ms)
    if backend is IndexBackend.FLAT_IP:
        return round(flat_ms, 2)
    if backend is IndexBackend.HNSW:
        return round(max(0.05, flat_ms * 0.12 + 0.05), 2)
    if backend is IndexBackend.IVF:
        return round(max(0.04, flat_ms * 0.08 + 0.03), 2)
    return round(max(0.03, flat_ms * 0.05 + 0.02), 2)


def expected_recall(backend: IndexBackend) -> float:
    if backend is IndexBackend.FLAT_IP:
        return 1.0
    if backend is IndexBackend.HNSW:
        return 0.99
    if backend is IndexBackend.IVF:
        return 0.90
    return 0.85


def comparison_table(
    *,
    catalog_size: int,
    dimension: int,
    params: BackendParams | None = None,
) -> tuple[BackendComparisonRow, ...]:
    params = params or BackendParams()
    rows: list[BackendComparisonRow] = []
    specs = [
        (
            IndexBackend.FLAT_IP,
            True,
            "1.00 (exact)",
            "Baseline",
            "Showrooms needing exact Top-K; catalogs <500k",
        ),
        (
            IndexBackend.HNSW,
            False,
            "~0.99",
            "Fast",
            "500k–1M+ when slight recall trade-off is OK",
        ),
        (
            IndexBackend.IVF,
            False,
            "~0.90",
            "Faster",
            "Large catalogs with tunable nprobe",
        ),
        (
            IndexBackend.IVF_PQ,
            False,
            "~0.85",
            "Fastest / lowest RAM",
            ">1M when system RAM cannot hold full vectors",
        ),
    ]
    for backend, exact, recall, speed, use in specs:
        mem = estimate_index_memory_mib(
            ntotal=catalog_size, dimension=dimension, backend=backend, params=params
        )
        rows.append(
            BackendComparisonRow(
                backend=backend_display_name(backend),
                exact=exact,
                recall=recall,
                memory=f"~{mem.total_mib:.0f} MiB",
                speed=speed,
                best_use_case=use,
            )
        )
    return tuple(rows)


_APPROX_WARNING = (
    "HNSW, IVF, and IVF-PQ are approximate indexes and may reduce recall "
    "compared with IndexFlatIP (exact). Production default stays FlatIP."
)


def advise_index_backend(
    *,
    catalog_size: int,
    current_backend: IndexBackend | str = IndexBackend.FLAT_IP,
    embedding_dimension: int = CURRENT_EMBEDDING_DIMENSION,
    available_ram_mib: float | None = None,
    cpu_cores: int | None = None,
    target_search_ms: float = 50.0,
    required_recall: float = 0.99,
    backend_params: BackendParams | None = None,
) -> IndexAdvice:
    """
    Compute a backend recommendation. Does not change settings or disk state.
    """
    params = backend_params or BackendParams()
    n = max(0, int(catalog_size))
    d = max(1, int(embedding_dimension))
    current = IndexBackend.parse(
        current_backend.value if isinstance(current_backend, IndexBackend) else current_backend
    )
    total_ram = available_ram_mib if available_ram_mib is not None else _total_ram_mib()
    usable = _usable_index_ram_mib(total_ram)
    cores = int(cpu_cores) if cpu_cores is not None else max(1, os.cpu_count() or 4)

    flat_mem = estimate_index_memory_mib(
        ntotal=n, dimension=d, backend=IndexBackend.FLAT_IP, params=params
    )
    hnsw_mem = estimate_index_memory_mib(
        ntotal=n, dimension=d, backend=IndexBackend.HNSW, params=params
    )
    pq_mem = estimate_index_memory_mib(
        ntotal=n, dimension=d, backend=IndexBackend.IVF_PQ, params=params
    )

    flat_fits = flat_mem.total_mib <= usable
    hnsw_fits = hnsw_mem.total_mib <= usable

    optional: IndexBackend | None = None
    optional_note = ""

    if n < _BAND_SMALL:
        recommended = IndexBackend.FLAT_IP
        reason = (
            f"Catalog size {n:,} is under 100k — IndexFlatIP is exact, "
            f"fits ~{flat_mem.total_mib:.0f} MiB, and stays well within "
            f"the ~{target_search_ms:.0f} ms latency target on typical CPUs."
        )
    elif n < _BAND_MEDIUM:
        recommended = IndexBackend.FLAT_IP
        optional = IndexBackend.HNSW
        reason = (
            f"Catalog size {n:,} (100k–500k) — keep IndexFlatIP for exact recall. "
            f"Estimated FlatIP RAM ~{flat_mem.total_mib:.0f} MiB "
            f"(budget ~{usable:.0f} MiB)."
        )
        optional_note = (
            "Optional: HNSW can lower latency but is approximate and may reduce recall."
        )
    elif n < _BAND_LARGE:
        recommended = IndexBackend.HNSW
        reason = (
            f"Catalog size {n:,} (500k–1M) — HNSW is recommended for search latency "
            f"while retaining high approximate recall (~{expected_recall(IndexBackend.HNSW):.2f}). "
            f"FlatIP remains available if you require exact Top-K and have RAM."
        )
        if flat_fits and required_recall >= 0.999:
            optional = IndexBackend.FLAT_IP
            optional_note = (
                "Exact mode: FlatIP still fits in RAM if you must keep recall = 1.0."
            )
    else:
        # >1M: IVF-PQ only when RAM cannot hold full-vector indexes.
        if not flat_fits and not hnsw_fits:
            recommended = IndexBackend.IVF_PQ
            reason = (
                f"Catalog size {n:,} (>1M) and estimated full-vector RAM "
                f"(FlatIP ~{flat_mem.total_mib:.0f} MiB / HNSW ~{hnsw_mem.total_mib:.0f} MiB) "
                f"exceeds the usable budget (~{usable:.0f} MiB). "
                f"IVF-PQ (~{pq_mem.total_mib:.0f} MiB) is recommended to fit memory — "
                f"it is approximate and may reduce recall."
            )
        else:
            recommended = IndexBackend.HNSW
            reason = (
                f"Catalog size {n:,} (>1M) with sufficient RAM for graph search — "
                f"HNSW recommended for latency. IVF-PQ is only advised when RAM is "
                f"insufficient for full-vector indexes "
                f"(FlatIP ~{flat_mem.total_mib:.0f} MiB, budget ~{usable:.0f} MiB)."
            )
            if flat_fits:
                optional = IndexBackend.FLAT_IP
                optional_note = (
                    "Exact FlatIP still fits in RAM but will be slower at this scale."
                )

    # If recommended ANN cannot meet required_recall and FlatIP fits, surface note.
    if (
        recommended is not IndexBackend.FLAT_IP
        and expected_recall(recommended) + 1e-9 < required_recall
        and flat_fits
    ):
        reason += (
            f" Note: required recall {required_recall:.2f} exceeds typical "
            f"{recommended.value} recall (~{expected_recall(recommended):.2f}); "
            f"consider FlatIP if accuracy is mandatory."
        )

    rec_mem = estimate_index_memory_mib(
        ntotal=n, dimension=d, backend=recommended, params=params
    )
    search_ms = estimate_search_latency_ms(
        recommended, catalog_size=n, dimension=d, cpu_cores=cores
    )
    if search_ms > target_search_ms and recommended is IndexBackend.FLAT_IP and n >= _BAND_SMALL:
        reason += (
            f" Estimated FlatIP search ~{search_ms:.1f} ms may exceed the "
            f"{target_search_ms:.0f} ms target; HNSW remains optional."
        )
        if optional is None and n < _BAND_MEDIUM:
            optional = IndexBackend.HNSW
            optional_note = (
                "Optional: HNSW for lower latency (approximate — may reduce recall)."
            )

    rebuild_required = recommended != current
    approx_warning = (
        _APPROX_WARNING if recommended is not IndexBackend.FLAT_IP else ""
    )

    return IndexAdvice(
        recommended_backend=recommended,
        reason=reason,
        estimated_ram_mib=rec_mem.total_mib,
        expected_recall=expected_recall(recommended),
        expected_search_ms=search_ms,
        rebuild_required=rebuild_required,
        optional_backend=optional,
        optional_note=optional_note,
        approximate_warning=approx_warning or _APPROX_WARNING,
        catalog_size=n,
        embedding_dimension=d,
        available_ram_mib=round(total_ram, 1) if total_ram is not None else None,
        cpu_cores=cores,
        current_backend=current,
        comparison=comparison_table(catalog_size=n, dimension=d, params=params),
    )
