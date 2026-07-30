"""
Startup diagnostics for TileVision AI v1.1+.

Logs hardware / software / catalog identity once at boot so field support
can diagnose showroom machines from a single log block.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from typing import Any

logger = logging.getLogger("tilevision.diagnostics")


def _rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        return None
    return None


def log_startup_diagnostics(info: dict[str, Any]) -> None:
    """Emit a structured startup identity block."""
    lines = [
        "=== TileVision Startup Diagnostics ===",
        f"App Version.......... {info.get('app_version', '?')}",
        f"Python............... {info.get('python', sys.version.split()[0])}",
        f"Platform............. {info.get('platform', platform.platform())}",
        f"CPU.................. {info.get('cpu', platform.processor() or platform.machine())}",
        f"CPU Count............ {info.get('cpu_count', os.cpu_count())}",
        f"RAM (RSS now)........ {info.get('rss_mb', _rss_mb())} MiB",
        f"Torch................ {info.get('torch', 'n/a')}",
        f"FAISS................ {info.get('faiss', 'n/a')}",
        f"Embedding Model...... {info.get('embedding_model', '?')}",
        f"Embedding Dim........ {info.get('embedding_dim', '?')}",
        f"Inference Device..... {info.get('device', '?')}",
        f"FAISS Type........... {info.get('faiss_type', '?')}",
        f"OpenMP Threads....... {info.get('omp_threads', '?')}",
        f"Catalog Size......... {info.get('catalog_size', '?')}",
        f"Database............. {info.get('database', '?')}",
        f"Index Path........... {info.get('index_path', '?')}",
        f"Profile Enabled...... {info.get('profile_enabled', '?')}",
        f"Log Level............ {info.get('log_level', '?')}",
    ]
    block = "\n".join(lines)
    logger.info("\n%s", block)
    print(block, flush=True)
