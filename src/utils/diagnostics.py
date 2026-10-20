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
        "query_warmup_ms": info.get("query_warmup_ms"),
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
        f"Model Warmup......... {report.get('model_warmup_ms', '?')} ms",
        f"Query Warmup......... {report.get('query_warmup_ms', '?')}"
        + (
            " ms"
            if isinstance(report.get("query_warmup_ms"), (int, float))
            else ""
        ),
        f"FAISS Warmup......... {report.get('faiss_warmup_ms', '?')} ms",
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


_SEAL_SUCCESS_MARKERS = (
    "Catalogue database encrypted at rest",
    "Catalogue database encrypted at rest.",
)
# Keys that must never appear in clipboard / paste-back diagnostics text.
_FORBIDDEN_DIAGNOSTIC_KEYS = frozenset(
    {
        "license_key",
        "licence_key",
        "password",
        "passwd",
        "secret",
        "api_key",
        "token",
        "TILEVISION_DB_PASSWORD",
    }
)


_UNKNOWN_SEAL_MESSAGE = (
    "unknown — quit the app once, relaunch, copy again"
)


def _previous_quit_sealed(log_path: Path) -> str:
    """
    Scan the tail of tilevision.log for the last seal-related shutdown line.

    Returns ``yes`` / ``no``, or ``_UNKNOWN_SEAL_MESSAGE`` when the log has
    no seal-related line (Copy Diagnostics is run while the app is open and
    reports on the *previous* quit).
    """
    if not log_path.is_file():
        return _UNKNOWN_SEAL_MESSAGE
    try:
        # Read last ~256 KiB — enough for several sessions without loading a
        # multi-megabyte rotating log into memory.
        data = log_path.read_bytes()
        if len(data) > 256 * 1024:
            data = data[-256 * 1024 :]
        text = data.decode("utf-8", errors="replace")
    except OSError:
        return _UNKNOWN_SEAL_MESSAGE

    sealed: bool | None = None
    for line in text.splitlines():
        if any(m in line for m in _SEAL_SUCCESS_MARKERS):
            sealed = True
        elif "Failed to encrypt catalogue database on exit" in line:
            sealed = False
        elif "Catalogue DB left unencrypted" in line:
            sealed = False
        elif "Skipping DB seal" in line:
            sealed = False
    if sealed is None:
        return _UNKNOWN_SEAL_MESSAGE
    return "yes" if sealed else "no"


def _tiles_db_present_now(database_path: str | None) -> str:
    """
    Whether the working plain ``tiles.db`` exists right now.

    While the app is open this is expected (decrypted working copy). After a
    clean quit only ``tiles.db.enc`` remains — check that in File Explorer
    before relaunching; this line must not imply a seal failure mid-session.
    """
    if not database_path:
        return "unknown"
    try:
        if database_path.endswith(".enc"):
            plain = Path(database_path[: -len(".enc")])
        else:
            plain = Path(database_path)
        return "yes" if plain.exists() else "no"
    except Exception:
        return "unknown"


def format_copy_diagnostics_text(info: dict[str, Any] | None = None) -> str:
    """
    Paste-friendly diagnostics for hardware smoke testing.

    Always includes app version, OS, log path, previous-quit seal status, and
    whether ``tiles.db`` is present now (expected while the app is open).
    Never includes licence keys or passwords (allowlisted fields only).
    """
    raw = dict(info or {})
    for bad in _FORBIDDEN_DIAGNOSTIC_KEYS:
        raw.pop(bad, None)
        # Also drop case-variants callers might pass.
        for key in list(raw.keys()):
            if str(key).casefold() == bad.casefold():
                raw.pop(key, None)

    report = collect_diagnostics_report(raw)
    from src.utils.logger import get_log_file_path

    log_path = get_log_file_path()
    db_path = report.get("database_path") or raw.get("database") or raw.get("database_path")
    seal_line = (
        'Previous quit sealed the database (log line '
        '"Catalogue database encrypted at rest"): '
        f"{_previous_quit_sealed(log_path)}"
    )
    lines = [
        "=== TileVision Copy Diagnostics ===",
        f"App version: {report.get('app_version', '?')}",
        f"OS / platform: {report.get('platform', '?')}",
        f"Log file: {log_path}",
        seal_line,
        f"tiles.db present now (expected while the app is open): "
        f"{_tiles_db_present_now(db_path)}",
        f"Database path: {db_path or '?'}",
        f"Catalog size: {report.get('catalog_size', '?')}",
        f"Inference device: {report.get('inference_device', '?')}",
        f"Generated at: {report.get('generated_at', '?')}",
    ]
    text = "\n".join(lines) + "\n"
    # Belt-and-suspenders: never let forbidden substrings from poisoned info
    # leak if a future field is added carelessly.
    for bad_key in _FORBIDDEN_DIAGNOSTIC_KEYS:
        needle = str(raw.get(bad_key, "") or "")
        if needle and needle in text:
            text = text.replace(needle, "***")
    return text
