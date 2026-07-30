"""
Startup / support diagnostics for TileVision AI v1.2 Enterprise.

Collects a structured report (CPU, RAM, GPU, library versions, catalog
identity) that can be logged at boot and exported as JSON from Settings.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("tilevision.diagnostics")


def _rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        return None
    return None


def _total_ram_mb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        return None
    return None


def collect_diagnostics_report(info: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Build a JSON-serializable diagnostics dict.

    ``info`` may supply already-known values (catalog size, paths, etc.)
    so callers avoid reloading heavy objects.
    """
    info = dict(info or {})

    torch_ver = info.get("torch")
    cuda_ver = info.get("cuda")
    gpu_name = info.get("gpu")
    if torch_ver is None or gpu_name is None:
        try:
            import torch

            torch_ver = torch_ver or getattr(torch, "__version__", "unknown")
            if torch.cuda.is_available():
                cuda_ver = cuda_ver or getattr(torch.version, "cuda", None)
                gpu_name = gpu_name or torch.cuda.get_device_name(0)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                gpu_name = gpu_name or "Apple MPS"
            else:
                gpu_name = gpu_name or "none"
        except Exception:
            torch_ver = torch_ver or "unavailable"
            gpu_name = gpu_name or "unavailable"

    faiss_ver = info.get("faiss")
    omp = info.get("omp_threads")
    if faiss_ver is None or omp is None:
        try:
            import faiss as _faiss

            faiss_ver = faiss_ver or getattr(_faiss, "__version__", "installed")
            omp = omp if omp is not None else int(_faiss.omp_get_max_threads())
        except Exception:
            faiss_ver = faiss_ver or "unavailable"
            omp = omp if omp is not None else 0

    sqlite_ver = info.get("sqlite")
    if sqlite_ver is None:
        try:
            import sqlite3

            sqlite_ver = sqlite3.sqlite_version
        except Exception:
            sqlite_ver = "unavailable"

    from src.version import APP_VERSION

    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "app_version": info.get("app_version", APP_VERSION),
        "python": info.get("python", sys.version.split()[0]),
        "platform": info.get("platform", platform.platform()),
        "cpu": info.get("cpu", platform.processor() or platform.machine()),
        "cpu_count": info.get("cpu_count", os.cpu_count()),
        "ram_total_mb": info.get("ram_total_mb", _total_ram_mb()),
        "ram_rss_mb": info.get("rss_mb", _rss_mb()),
        "gpu": gpu_name,
        "cuda": cuda_ver,
        "torch": torch_ver,
        "faiss": faiss_ver,
        "sqlite": sqlite_ver,
        "openmp_threads": omp,
        "embedding_model": info.get("embedding_model"),
        "embedding_dimension": info.get("embedding_dim") or info.get("embedding_dimension"),
        "inference_device": info.get("device") or info.get("inference_device"),
        "index_type": info.get("faiss_type") or info.get("index_type"),
        "index_backend": info.get("index_backend"),
        "catalog_size": info.get("catalog_size"),
        "database_path": info.get("database") or info.get("database_path"),
        "index_path": info.get("index_path"),
        "profile_enabled": info.get("profile_enabled"),
        "log_level": info.get("log_level"),
        "model_warmup_ms": info.get("model_warmup_ms"),
        "faiss_warmup_ms": info.get("faiss_warmup_ms"),
        "compatibility": info.get("compatibility"),
    }
    # Drop pure-None keys that callers never supplied (keep structural keys).
    return report


def log_startup_diagnostics(info: dict[str, Any]) -> None:
    """Emit a structured startup identity block."""
    report = collect_diagnostics_report(info)
    lines = [
        "=== TileVision Startup Diagnostics ===",
        f"App Version.......... {report.get('app_version', '?')}",
        f"Python............... {report.get('python', '?')}",
        f"Platform............. {report.get('platform', '?')}",
        f"CPU.................. {report.get('cpu', '?')}",
        f"CPU Count............ {report.get('cpu_count', '?')}",
        f"RAM Total............ {report.get('ram_total_mb', '?')} MiB",
        f"RAM (RSS now)........ {report.get('ram_rss_mb', '?')} MiB",
        f"GPU.................. {report.get('gpu', '?')}",
        f"Torch................ {report.get('torch', 'n/a')}",
        f"FAISS................ {report.get('faiss', 'n/a')}",
        f"SQLite............... {report.get('sqlite', 'n/a')}",
        f"Embedding Model...... {report.get('embedding_model', '?')}",
        f"Embedding Dim........ {report.get('embedding_dimension', '?')}",
        f"Inference Device..... {report.get('inference_device', '?')}",
        f"Index Backend........ {report.get('index_backend', '?')}",
        f"FAISS Type........... {report.get('index_type', '?')}",
        f"OpenMP Threads....... {report.get('openmp_threads', '?')}",
        f"Catalog Size......... {report.get('catalog_size', '?')}",
        f"Database............. {report.get('database_path', '?')}",
        f"Index Path........... {report.get('index_path', '?')}",
        f"Profile Enabled...... {report.get('profile_enabled', '?')}",
        f"Log Level............ {report.get('log_level', '?')}",
    ]
    block = "\n".join(lines)
    logger.info("\n%s", block)
    print(block, flush=True)


def export_diagnostics_json(
    destination: str | Path,
    info: dict[str, Any] | None = None,
) -> Path:
    """Write diagnostics report to ``destination`` as pretty JSON."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = collect_diagnostics_report(info)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Diagnostics report exported to %s", path)
    return path
