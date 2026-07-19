"""
Cross-platform helpers for TileVision AI.

Centralizes OS detection used by GPU checks, UI defaults, and setup docs
so Mac/Linux/Windows behavior stays consistent.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from src.utils.brand_assets import APP_ICON_PATH, APP_ICON_PNG_PATH


def is_windows() -> bool:
    return sys.platform == "win32"


def is_macos() -> bool:
    return sys.platform == "darwin"


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def _run_command(args: Sequence[str], *, timeout: float = 15.0) -> str | None:
    try:
        kwargs: dict = {
            "capture_output": True,
            "text": True,
            "timeout": timeout,
            "check": False,
        }
        if is_windows():
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            kwargs["startupinfo"] = startupinfo

        result = subprocess.run(list(args), **kwargs)
        if result.returncode != 0:
            return None
        output = (result.stdout or "").strip()
        return output or None
    except Exception:
        return None


def detect_windows_graphics() -> list[str]:
    output = _run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object -ExpandProperty Name",
        ]
    )
    if not output:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def detect_macos_graphics() -> list[str]:
    output = _run_command(["system_profiler", "SPDisplaysDataType"])
    if not output:
        return []

    adapters: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Chipset Model:"):
            adapters.append(stripped.split(":", 1)[-1].strip())
        elif stripped.startswith("Model:") and "Displays" not in stripped:
            adapters.append(stripped.split(":", 1)[-1].strip())
    return adapters


def detect_linux_graphics() -> list[str]:
    if shutil.which("nvidia-smi"):
        output = _run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        if output:
            return [line.strip() for line in output.splitlines() if line.strip()]

    if shutil.which("lspci"):
        output = _run_command(["lspci", "-mm"])
        if output:
            adapters: list[str] = []
            for line in output.splitlines():
                parts = [part.strip('"') for part in line.split('"') if part.strip()]
                if len(parts) >= 4 and parts[2] == "Display controller":
                    adapters.append(parts[3])
            if adapters:
                return adapters

    return []


def detect_display_adapters() -> list[str]:
    """Return human-readable graphics adapter names for the current OS."""
    if is_windows():
        return detect_windows_graphics()
    if is_macos():
        return detect_macos_graphics()
    return detect_linux_graphics()


def has_nvidia_gpu() -> bool:
    """Return True when an NVIDIA GPU appears to be present."""
    if shutil.which("nvidia-smi"):
        return True
    return any("nvidia" in name.lower() for name in detect_display_adapters())


def cuda_pytorch_install_command() -> str:
    """Return a platform-appropriate hint for installing CUDA PyTorch."""
    if is_windows():
        return "powershell -ExecutionPolicy Bypass -File scripts/install_pytorch_cuda.ps1"
    if is_linux():
        return "bash scripts/install_pytorch_cuda.sh"
    return "Use the standard PyTorch wheel — Apple Silicon uses MPS acceleration automatically."


def default_ui_font_family() -> str:
    if is_macos():
        return ".AppleSystemUIFont"
    if is_linux():
        return "Ubuntu"
    return "Segoe UI"


def app_icon_path() -> Path | None:
    """Prefer PNG on macOS/Linux; ICO on Windows when available."""
    if is_windows() and APP_ICON_PATH.exists():
        return APP_ICON_PATH
    if APP_ICON_PNG_PATH.exists():
        return APP_ICON_PNG_PATH
    if APP_ICON_PATH.exists():
        return APP_ICON_PATH
    return None


def user_data_dir_hint() -> str:
    return str(Path.home() / ".tilevision_ai")
