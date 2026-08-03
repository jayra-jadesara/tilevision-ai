"""
Search Optimization Engine for TileVision AI.

Single authority for choosing the FAISS search backend. Customers never pick
backends — this engine inspects the real machine + catalog, optionally runs a
lightweight latency probe, and decides with accuracy first.

Priority: Accuracy → Reliability → Consistency → Speed → Memory.

IndexFlatIP remains selected whenever measured memory shows it fits.
Approximate backends are chosen only when FlatIP cannot fit in measured
available RAM (never merely because they exist).
"""

from __future__ import annotations

import logging
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from src.ai.feature_versions import CURRENT_EMBEDDING_DIMENSION
from src.ai.index_backends import (
    BackendParams,
    IndexBackend,
    create_empty_index,
    estimate_index_memory_mib,
)
from src.ai.index_metadata import read_index_metadata
from src.version import APP_VERSION

logger = logging.getLogger("tilevision.ai.search_optimization_engine")


@dataclass(frozen=True, slots=True)
class SearchEnvironmentSnapshot:
    """Measured environment used for a decision (no guessed catalog sizes)."""

    catalog_size: int
    embedding_dimension: int
    available_ram_mib: float | None
    total_ram_mib: float | None
    process_rss_mib: float | None
    model_footprint_mib: float | None
    usable_index_ram_mib: float
    cpu_cores: int
    cpu_model: str
    cpu_architecture: str
    operating_system: str
    index_path: str
    index_file_size_mib: float | None
    existing_backend: str
    existing_index_version: str
    app_version: str
    measured_search_latency_ms: float | None
    estimated_flat_ip_mib: float
    estimated_hnsw_mib: float
    estimated_ivf_pq_mib: float
    benchmark_ran: bool
    benchmark_detail: str


@dataclass(frozen=True, slots=True)
class OptimizationDecision:
    """Result of analyze_and_decide()."""

    selected_backend: IndexBackend
    rebuild_required: bool
    status: str  # Optimized | Needs Optimization
    search_health: str  # Excellent | Good | Fair
    index_status: str  # Healthy | Needs rebuild
    customer_summary: str  # no FAISS/backend jargon
    technical_reason: str  # developer / logs
    snapshot: SearchEnvironmentSnapshot
    decided_at: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_backend": self.selected_backend.value,
            "rebuild_required": self.rebuild_required,
            "status": self.status,
            "search_health": self.search_health,
            "index_status": self.index_status,
            "customer_summary": self.customer_summary,
            "technical_reason": self.technical_reason,
            "decided_at": self.decided_at,
            "snapshot": asdict(self.snapshot),
            "diagnostics": self.diagnostics,
        }


def is_developer_mode() -> bool:
    """Same gate as licensing — only TILEVISION_DEV_MODE=1."""
    return os.environ.get("TILEVISION_DEV_MODE") == "1"


def format_tile_count(count: int) -> str:
    """Customer-facing catalog label from a live count."""
    n = max(0, int(count))
    label = "Tile" if n == 1 else "Tiles"
    return f"{n:,} {label}"


class SearchOptimizationEngine:
    """
    Owns backend selection for production.

    Inject measured values via constructor call args / analyze() parameters so
    unit tests can supply fixtures without hardcoding production thresholds.
    """

    def __init__(
        self,
        *,
        index_path: str | Path,
        embedding_dimension: int = CURRENT_EMBEDDING_DIMENSION,
        backend_params: BackendParams | None = None,
        model_dir: str | Path | None = None,
        vector_index: Any | None = None,
        catalog_count_provider: Optional[Callable[[], int]] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._index_path = Path(index_path)
        self._dimension = max(1, int(embedding_dimension))
        self._params = backend_params or BackendParams()
        self._model_dir = Path(model_dir) if model_dir else None
        self._vector_index = vector_index
        self._catalog_count_provider = catalog_count_provider
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._last_decision: OptimizationDecision | None = None

    @property
    def last_decision(self) -> OptimizationDecision | None:
        return self._last_decision

    def analyze_and_decide(
        self,
        *,
        catalog_size: int | None = None,
        current_backend: IndexBackend | str | None = None,
        run_benchmark: bool = True,
    ) -> OptimizationDecision:
        snapshot = self.capture_environment(
            catalog_size=catalog_size,
            current_backend=current_backend,
            run_benchmark=run_benchmark,
        )
        decision = self._decide(snapshot)
        self._last_decision = decision
        logger.info(
            "[SOE] decision backend=%s rebuild=%s health=%s catalog=%s "
            "usable_ram=%.1f flat_est=%.1f measured_ms=%s",
            decision.selected_backend.value,
            decision.rebuild_required,
            decision.search_health,
            snapshot.catalog_size,
            snapshot.usable_index_ram_mib,
            snapshot.estimated_flat_ip_mib,
            snapshot.measured_search_latency_ms,
        )
        return decision

    def apply_to_settings(self, settings: Any, decision: OptimizationDecision) -> bool:
        """
        Persist decision into AppSettings. Returns True if configured backend changed.
        """
        previous = str(getattr(settings, "index_backend", "flat_ip"))
        settings.index_backend = decision.selected_backend.value
        settings.search_engine_mode = "automatic"
        settings.search_optimization_status = decision.status
        settings.search_health = decision.search_health
        settings.index_health_status = decision.index_status
        settings.last_search_optimization_at = decision.decided_at
        settings.last_optimization_catalog_size = decision.snapshot.catalog_size
        settings.last_optimization_app_version = APP_VERSION
        settings.last_optimization_summary = decision.customer_summary
        return previous != decision.selected_backend.value

    def capture_environment(
        self,
        *,
        catalog_size: int | None = None,
        current_backend: IndexBackend | str | None = None,
        run_benchmark: bool = True,
    ) -> SearchEnvironmentSnapshot:
        n = catalog_size
        if n is None:
            n = self._resolve_catalog_size()
        n = max(0, int(n))

        total_ram, available_ram = self._measure_ram_mib()
        process_rss = self._measure_process_rss_mib()
        model_footprint = self._measure_model_footprint_mib()
        usable = self._compute_usable_index_ram_mib(
            available_ram_mib=available_ram,
            process_rss_mib=process_rss,
            model_footprint_mib=model_footprint,
        )

        flat_est = estimate_index_memory_mib(
            ntotal=n, dimension=self._dimension, backend=IndexBackend.FLAT_IP, params=self._params
        ).total_mib
        hnsw_est = estimate_index_memory_mib(
            ntotal=n, dimension=self._dimension, backend=IndexBackend.HNSW, params=self._params
        ).total_mib
        pq_est = estimate_index_memory_mib(
            ntotal=n, dimension=self._dimension, backend=IndexBackend.IVF_PQ, params=self._params
        ).total_mib

        existing_backend, existing_version = self._read_existing_index_info()
        # existing_backend is on-disk (or live active). Do not overwrite with settings.
        if current_backend is not None:
            # Used only for diagnostics; decision compares against on-disk.
            _ = IndexBackend.parse(
                current_backend.value
                if isinstance(current_backend, IndexBackend)
                else current_backend
            ).value

        index_size = None
        if self._index_path.is_file():
            try:
                index_size = self._index_path.stat().st_size / (1024.0 * 1024.0)
            except OSError:
                index_size = None

        measured_ms = None
        benchmark_ran = False
        benchmark_detail = "skipped"
        if run_benchmark:
            measured_ms, benchmark_detail = self._run_latency_benchmark(catalog_size=n)
            benchmark_ran = measured_ms is not None

        return SearchEnvironmentSnapshot(
            catalog_size=n,
            embedding_dimension=self._dimension,
            available_ram_mib=available_ram,
            total_ram_mib=total_ram,
            process_rss_mib=process_rss,
            model_footprint_mib=model_footprint,
            usable_index_ram_mib=usable,
            cpu_cores=max(1, os.cpu_count() or 1),
            cpu_model=platform.processor() or platform.machine() or "unknown",
            cpu_architecture=platform.machine() or "unknown",
            operating_system=f"{platform.system()} {platform.release()}".strip(),
            index_path=str(self._index_path),
            index_file_size_mib=index_size,
            existing_backend=existing_backend or IndexBackend.FLAT_IP.value,
            existing_index_version=existing_version,
            app_version=APP_VERSION,
            measured_search_latency_ms=measured_ms,
            estimated_flat_ip_mib=float(flat_est),
            estimated_hnsw_mib=float(hnsw_est),
            estimated_ivf_pq_mib=float(pq_est),
            benchmark_ran=benchmark_ran,
            benchmark_detail=benchmark_detail,
        )

    def _decide(self, snapshot: SearchEnvironmentSnapshot) -> OptimizationDecision:
        usable = snapshot.usable_index_ram_mib
        flat_fits = snapshot.estimated_flat_ip_mib <= usable
        hnsw_fits = snapshot.estimated_hnsw_mib <= usable
        pq_fits = snapshot.estimated_ivf_pq_mib <= usable

        # Accuracy first: FlatIP whenever measured memory says it fits.
        if flat_fits or snapshot.catalog_size <= 0:
            selected = IndexBackend.FLAT_IP
            technical = (
                f"IndexFlatIP selected — estimated {snapshot.estimated_flat_ip_mib:.1f} MiB "
                f"fits in measured usable index RAM {usable:.1f} MiB "
                f"(available={snapshot.available_ram_mib}, rss={snapshot.process_rss_mib}, "
                f"model={snapshot.model_footprint_mib}). Exact Top-K preserved."
            )
            customer = (
                "Search is set for exact, reliable matching based on this computer "
                f"and your current catalog ({format_tile_count(snapshot.catalog_size)})."
            )
            health = "Excellent"
        elif hnsw_fits:
            selected = IndexBackend.HNSW
            technical = (
                f"HNSW selected — FlatIP estimate {snapshot.estimated_flat_ip_mib:.1f} MiB "
                f"exceeds usable {usable:.1f} MiB; HNSW estimate "
                f"{snapshot.estimated_hnsw_mib:.1f} MiB fits. Accuracy priority deferred "
                "only due to measured memory pressure."
            )
            customer = (
                "Search was tuned automatically for this computer’s available memory "
                f"while keeping results as accurate as possible "
                f"({format_tile_count(snapshot.catalog_size)})."
            )
            health = "Good"
        elif pq_fits:
            selected = IndexBackend.IVF_PQ
            technical = (
                f"IVF-PQ selected — FlatIP/HNSW estimates exceed usable RAM "
                f"{usable:.1f} MiB; IVF-PQ estimate {snapshot.estimated_ivf_pq_mib:.1f} MiB fits."
            )
            customer = (
                "Search was optimized automatically for limited memory on this computer. "
                f"Catalog: {format_tile_count(snapshot.catalog_size)}."
            )
            health = "Fair"
        else:
            # Still prefer exact search; do not invent an unsafe switch.
            selected = IndexBackend.FLAT_IP
            technical = (
                f"IndexFlatIP retained despite memory pressure "
                f"(flat={snapshot.estimated_flat_ip_mib:.1f} MiB, usable={usable:.1f} MiB) "
                "to preserve accuracy."
            )
            customer = (
                "Search stays on exact matching to protect result quality. "
                "If the app feels slow, free memory or rebuild after closing other programs."
            )
            health = "Fair"

        existing = IndexBackend.parse(snapshot.existing_backend)
        rebuild_required = existing != selected and snapshot.catalog_size > 0
        if rebuild_required:
            status = "Needs Optimization"
            index_status = "Needs rebuild"
            customer = (
                "Search optimization is ready. TileVision will rebuild the search index "
                "automatically so results stay fast and accurate — no action needed."
            )
        else:
            status = "Optimized"
            index_status = "Healthy"

        decided_at = self._now_fn().astimezone().strftime("%Y-%m-%d %H:%M:%S")
        diagnostics = {
            "selected_backend": selected.value,
            "existing_backend": existing.value,
            "flat_fits": flat_fits,
            "hnsw_fits": hnsw_fits,
            "ivf_pq_fits": pq_fits,
            "usable_index_ram_mib": usable,
            "measured_search_latency_ms": snapshot.measured_search_latency_ms,
            "benchmark_detail": snapshot.benchmark_detail,
            "operating_system": snapshot.operating_system,
            "cpu_architecture": snapshot.cpu_architecture,
            "cpu_model": snapshot.cpu_model,
            "cpu_cores": snapshot.cpu_cores,
            "catalog_size": snapshot.catalog_size,
            "embedding_dimension": snapshot.embedding_dimension,
            "index_file_size_mib": snapshot.index_file_size_mib,
            "developer_mode": is_developer_mode(),
        }
        return OptimizationDecision(
            selected_backend=selected,
            rebuild_required=rebuild_required,
            status=status,
            search_health=health,
            index_status=index_status,
            customer_summary=customer,
            technical_reason=technical,
            snapshot=snapshot,
            decided_at=decided_at,
            diagnostics=diagnostics,
        )

    def _resolve_catalog_size(self) -> int:
        if self._vector_index is not None:
            try:
                count = int(self._vector_index.get_total_count())
                if count > 0:
                    return count
            except Exception as exc:
                logger.debug("vector_index catalog count failed: %s", exc)
        if self._catalog_count_provider is not None:
            try:
                return max(0, int(self._catalog_count_provider() or 0))
            except Exception as exc:
                logger.debug("catalog_count_provider failed: %s", exc)
        return 0

    def _read_existing_index_info(self) -> tuple[str, str]:
        backend = IndexBackend.FLAT_IP.value
        version = ""
        if self._vector_index is not None:
            try:
                backend = self._vector_index.active_backend().value
            except Exception:
                try:
                    backend = self._vector_index.configured_backend().value
                except Exception:
                    pass
        meta = read_index_metadata(self._index_path)
        if meta is not None:
            if meta.index_backend:
                backend = str(meta.index_backend)
            version = str(getattr(meta, "app_version", "") or "")
        return backend, version

    @staticmethod
    def _measure_ram_mib() -> tuple[float | None, float | None]:
        """Return (total_mib, available_mib) from the OS when possible."""
        system = platform.system()
        if system == "Linux":
            total = available = None
            try:
                with open("/proc/meminfo", encoding="utf-8") as handle:
                    data = {}
                    for line in handle:
                        parts = line.split()
                        if len(parts) >= 2:
                            data[parts[0].rstrip(":")] = int(parts[1]) / 1024.0
                total = data.get("MemTotal")
                available = data.get("MemAvailable", data.get("MemFree"))
            except Exception:
                return None, None
            return total, available
        try:
            import psutil  # type: ignore

            vm = psutil.virtual_memory()
            return vm.total / (1024.0 * 1024.0), vm.available / (1024.0 * 1024.0)
        except Exception:
            pass
        if system == "Darwin":
            try:
                import subprocess

                total_bytes = int(
                    subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
                )
                # page-free approx via vm_stat
                vm = subprocess.check_output(["vm_stat"], text=True)
                page_size = 4096
                free_pages = 0
                for line in vm.splitlines():
                    if line.startswith("Pages free") or line.startswith("Pages speculative"):
                        free_pages += int(line.split(":")[1].strip().rstrip("."))
                    if "page size of" in line:
                        try:
                            page_size = int(line.split("page size of")[1].split()[0])
                        except Exception:
                            pass
                available = (free_pages * page_size) / (1024.0 * 1024.0)
                return total_bytes / (1024.0 * 1024.0), available
            except Exception:
                return None, None
        if system == "Windows":
            try:
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return (
                    stat.ullTotalPhys / (1024.0 * 1024.0),
                    stat.ullAvailPhys / (1024.0 * 1024.0),
                )
            except Exception:
                return None, None
        return None, None

    @staticmethod
    def _measure_process_rss_mib() -> float | None:
        try:
            import psutil  # type: ignore

            return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            pass
        if platform.system() == "Linux":
            try:
                with open("/proc/self/status", encoding="utf-8") as handle:
                    for line in handle:
                        if line.startswith("VmRSS:"):
                            return int(line.split()[1]) / 1024.0
            except Exception:
                return None
        return None

    def _measure_model_footprint_mib(self) -> float | None:
        root = self._model_dir
        if root is None:
            try:
                from src.ai.model_paths import bundled_model_dir, resolve_dinov2_model_source

                source, _local = resolve_dinov2_model_source()
                candidate = Path(source)
                root = candidate if candidate.is_dir() else bundled_model_dir()
            except Exception:
                root = None
        if root is None or not Path(root).exists():
            return None
        total = 0
        try:
            for path in Path(root).rglob("*"):
                if path.is_file():
                    total += path.stat().st_size
        except OSError:
            return None
        return total / (1024.0 * 1024.0)

    @staticmethod
    def _compute_usable_index_ram_mib(
        *,
        available_ram_mib: float | None,
        process_rss_mib: float | None,
        model_footprint_mib: float | None,
    ) -> float:
        """
        Derive index RAM budget from measured free memory.

        Uses available RAM when present. Model footprint is already largely
        reflected in process RSS after warm-up; when RSS is missing, subtract
        on-disk model size as a conservative stand-in.
        """
        if available_ram_mib is not None and available_ram_mib > 0:
            # Available is already "free-ish"; keep a relative reserve of 10%
            # of that measured pool (not a fixed GiB threshold).
            reserve = available_ram_mib * 0.10
            usable = max(0.0, available_ram_mib - reserve)
            return usable
        # Fallback when OS will not report availability: allow FlatIP unless
        # estimates are enormous relative to process size (still measurement-tied).
        rss = process_rss_mib or 0.0
        model = model_footprint_mib or 0.0
        return max(512.0, rss + model)

    def _run_latency_benchmark(self, *, catalog_size: int) -> tuple[float | None, str]:
        """
        Lightweight search timing probe. Returns (median_ms, detail).

        Uses the live FAISS index when loaded; otherwise builds a temporary
        FlatIP with a sample of random unit vectors sized to the catalog
        (capped for speed). Does not change production index files.
        """
        try:
            import faiss  # noqa: F401
        except Exception as exc:
            return None, f"benchmark unavailable: faiss import failed ({exc})"

        d = self._dimension
        rng = np.random.default_rng(0)

        live = self._vector_index
        if live is not None:
            try:
                ntotal = int(live.get_total_count())
            except Exception:
                ntotal = 0
            if ntotal > 0:
                try:
                    queries = 5
                    q = rng.standard_normal((queries, d), dtype=np.float32)
                    # L2 normalize for IP indexes
                    norms = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
                    q = q / norms
                    times = []
                    for i in range(queries):
                        t0 = time.perf_counter()
                        live.search_vectors(q[i], top_k=min(10, ntotal))
                        times.append((time.perf_counter() - t0) * 1000.0)
                    median = float(np.median(times))
                    return median, f"live_index n={ntotal} queries={queries}"
                except Exception as exc:
                    return None, f"live benchmark failed: {exc}"

        # Synthetic probe — sample size tracks catalog but stays practical.
        sample_n = min(max(catalog_size, 0), 2_000)
        if sample_n <= 0:
            return None, "no vectors to benchmark"
        try:
            index = create_empty_index(
                dimension=d, backend=IndexBackend.FLAT_IP, params=self._params
            )
            vectors = rng.standard_normal((sample_n, d), dtype=np.float32)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
            ids = np.arange(sample_n, dtype=np.int64)
            index.add_with_ids(vectors, ids)
            q = rng.standard_normal((8, d), dtype=np.float32)
            q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
            times = []
            for i in range(8):
                t0 = time.perf_counter()
                index.search(q[i : i + 1], 10)
                times.append((time.perf_counter() - t0) * 1000.0)
            return float(np.median(times)), f"synthetic_flat_ip n={sample_n}"
        except Exception as exc:
            return None, f"synthetic benchmark failed: {exc}"
