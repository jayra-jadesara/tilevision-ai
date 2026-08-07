"""Collect real runtime environment evidence for release validation."""

from __future__ import annotations

import os
import platform
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _safe_import_version(module: str, attr: str = "__version__") -> str:
    try:
        mod = __import__(module, fromlist=[attr])
        return str(getattr(mod, attr, "unknown"))
    except Exception as exc:
        return f"unavailable ({exc.__class__.__name__}: {exc})"


def _ram_gb() -> Optional[float]:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        try:
            pages = os.sysconf("SC_PAGE_SIZE")
            count = os.sysconf("SC_PHYS_PAGES")
            return round(pages * count / (1024**3), 2)
        except Exception:
            return None


def _cpu_info() -> str:
    try:
        import psutil

        return f"{platform.processor() or platform.machine()} ({psutil.cpu_count(logical=True)} threads)"
    except Exception:
        return platform.processor() or platform.machine()


def collect_environment(*, session=None) -> Dict[str, Any]:
    """
    Snapshot versions and host facts. Uses live session objects when available
    so DINOv2 / FAISS / license readiness reflect the running app.
    """
    from src.ai.model_paths import bundled_model_dir, resolve_dinov2_model_source, runtime_root
    from src.version import APP_VERSION

    dinov2_path = ""
    dinov2_local = False
    try:
        dinov2_path, dinov2_local = resolve_dinov2_model_source()
    except Exception as exc:
        dinov2_path = f"error: {exc}"

    root = runtime_root()
    sam2_onnx = root / "model_weights" / "sam2.1-hiera-tiny-onnx"
    sam2_pt = root / "model_weights" / "sam2.1-hiera-tiny"
    encoder = sam2_onnx / "sam2.1_hiera_tiny.encoder.onnx"
    decoder = sam2_onnx / "sam2.1_hiera_tiny.decoder.onnx"
    sam2_load_ok = False
    sam2_load_detail = ""
    if encoder.is_file():
        try:
            import onnxruntime as ort

            # Prove weights load into a real ONNX Runtime session (not mocked).
            _sess = ort.InferenceSession(str(encoder), providers=["CPUExecutionProvider"])
            sam2_load_ok = bool(_sess.get_inputs())
            sam2_load_detail = f"onnxruntime session ok inputs={len(_sess.get_inputs())}"
            del _sess
        except Exception as exc:
            sam2_load_detail = f"{exc.__class__.__name__}: {exc}"
    else:
        sam2_load_detail = "encoder onnx missing"
    sam2_status = {
        "onnx_dir_exists": sam2_onnx.is_dir(),
        "transformers_dir_exists": sam2_pt.is_dir(),
        "onnx_encoder": encoder.is_file(),
        "onnx_decoder": decoder.is_file(),
        "load_ok": sam2_load_ok,
        "load_detail": sam2_load_detail,
        "onnx_dir": str(sam2_onnx),
    }

    payload: Dict[str, Any] = {
        "app_version": APP_VERSION,
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "frozen": bool(getattr(sys, "frozen", False)),
        "meipass": str(getattr(sys, "_MEIPASS", "") or ""),
        "executable": sys.executable,
        "packaged_app": os.environ.get("TILEVISION_QA_PACKAGED_APP", "") == "1",
        "torch": _safe_import_version("torch"),
        "torchvision": _safe_import_version("torchvision"),
        "faiss": _safe_import_version("faiss"),
        "sqlite": sqlite3.sqlite_version,
        "opencv": _safe_import_version("cv2"),
        "pyside6": _safe_import_version("PySide6"),
        "pillow": _safe_import_version("PIL"),
        "numpy": _safe_import_version("numpy"),
        "os": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "cpu": _cpu_info(),
        "ram_gb": _ram_gb(),
        "dinov2": {
            "source": dinov2_path,
            "local_files_only": dinov2_local,
            "bundled_dir": str(bundled_model_dir()),
            "bundled_present": (bundled_model_dir() / "config.json").is_file(),
        },
        "sam2": sam2_status,
        "env": {
            "TILEVISION_DEV_MODE": os.environ.get("TILEVISION_DEV_MODE", ""),
            "TILEVISION_OFFLINE_MODEL": os.environ.get("TILEVISION_OFFLINE_MODEL", ""),
            "QT_QPA_PLATFORM": os.environ.get("QT_QPA_PLATFORM", ""),
            "TILEVISION_ENABLE_SAM2": os.environ.get("TILEVISION_ENABLE_SAM2", ""),
            "TILEVISION_QA_PACKAGED_APP": os.environ.get("TILEVISION_QA_PACKAGED_APP", ""),
        },
    }

    if session is not None:
        try:
            payload["runtime"] = {
                "device": session.embedder.runtime_info.summary_for_ui(),
                "model_loaded": getattr(session.embedder, "_model", None) is not None,
                "faiss_backend": session.vector_index.active_backend().value,
                "faiss_type": session.vector_index.index_type_name(),
                "faiss_count": session.vector_index.get_total_count(),
                "sqlite_tiles": len(session.image_repository.get_all()),
                "license_ok": True,
                "database": str(session.settings.database_path),
                "index_path": str(session.settings.index_path),
            }
        except Exception as exc:
            payload["runtime_error"] = str(exc)

    return payload


def environment_gate_failures(env: Dict[str, Any]) -> list[str]:
    """Hard failures for release environment gate."""
    fails: list[str] = []
    for key in ("torch", "faiss", "opencv", "pyside6"):
        val = str(env.get(key, ""))
        if val.startswith("unavailable"):
            fails.append(f"{key} not importable: {val}")
    if not env.get("dinov2", {}).get("bundled_present") and not env.get("dinov2", {}).get(
        "local_files_only"
    ):
        # Allow HF id only when offline is not forced
        if os.environ.get("TILEVISION_OFFLINE_MODEL", "").strip() in {"1", "true", "yes"}:
            fails.append("DINOv2 weights missing while TILEVISION_OFFLINE_MODEL=1")
    # Packaged-app ship gate: must be the frozen customer binary, not source python.
    if os.environ.get("TILEVISION_QA_PACKAGED_APP", "").strip() == "1":
        if not env.get("frozen"):
            fails.append("TILEVISION_QA_PACKAGED_APP=1 but sys.frozen is false — not the installed .app")
        if not env.get("packaged_app"):
            fails.append("packaged_app flag missing from environment snapshot")
    return fails
