"""
GPU runtime detection for TileVision AI.

Reports whether CUDA is available and why the app falls back to CPU.
Used at startup, in Settings, and by dev_tools/check_gpu.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import torch

logger = logging.getLogger("tilevision.ai.gpu_info")

DevicePreference = Literal["auto", "cuda", "cpu"]


@dataclass(frozen=True, slots=True)
class GpuRuntimeInfo:
    """Snapshot of PyTorch/CUDA availability at runtime."""

    is_available: bool
    active_device: str
    device_name: str
    device_count: int
    torch_version: str
    cuda_version: str | None
    vram_gb: float | None
    cpu_fallback_reason: str

    @property
    def using_gpu(self) -> bool:
        return self.active_device == "cuda"

    def summary_for_ui(self) -> str:
        if self.using_gpu:
            vram = f", {self.vram_gb:.1f} GB VRAM" if self.vram_gb else ""
            return f"{self.device_name}{vram}"
        return f"CPU — {self.cpu_fallback_reason}"

    def summary_for_log(self) -> str:
        if self.using_gpu:
            return (
                f"GPU active: {self.device_name} "
                f"(CUDA {self.cuda_version}, torch {self.torch_version})"
            )
        return f"CPU mode: {self.cpu_fallback_reason} (torch {self.torch_version})"


def _detect_windows_graphics() -> list[str]:
    """Return display adapter names on Windows (empty on other OS)."""
    import subprocess
    import sys

    if sys.platform != "win32":
        return []

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object -ExpandProperty Name",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _has_nvidia_adapter(adapters: list[str]) -> bool:
    return any("nvidia" in name.lower() for name in adapters)


def _cpu_torch_reason() -> str:
    adapters = _detect_windows_graphics()
    if adapters and not _has_nvidia_adapter(adapters):
        names = ", ".join(adapters[:2])
        return f"no NVIDIA GPU detected ({names}) — CUDA requires NVIDIA hardware"

    version = torch.__version__.lower()
    if "+cpu" in version or "cpu" in version.split("+")[-1:]:
        return "CPU-only PyTorch — install CUDA wheel after NVIDIA driver"
    if not torch.cuda.is_available():
        return "CUDA not available — install NVIDIA driver and CUDA PyTorch"
    return "CUDA unavailable"


def detect_gpu_runtime(*, preference: DevicePreference = "auto") -> GpuRuntimeInfo:
    """Inspect torch/CUDA and resolve the active inference device."""
    pref = (preference or "auto").lower()
    if pref not in ("auto", "cuda", "cpu"):
        pref = "auto"

    cuda_available = torch.cuda.is_available()
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    device_name = ""
    vram_gb: float | None = None
    cuda_version = torch.version.cuda

    if cuda_available and device_count > 0:
        device_name = torch.cuda.get_device_name(0)
        try:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
        except Exception:
            vram_gb = None

    if pref == "cpu":
        active = "cpu"
        reason = "forced by inference_device=cpu setting"
    elif pref == "cuda":
        if cuda_available:
            active = "cuda"
            reason = ""
        else:
            active = "cpu"
            reason = _cpu_torch_reason()
    else:
        if cuda_available:
            active = "cuda"
            reason = ""
        else:
            active = "cpu"
            reason = _cpu_torch_reason()

    return GpuRuntimeInfo(
        is_available=cuda_available,
        active_device=active,
        device_name=device_name,
        device_count=device_count,
        torch_version=torch.__version__,
        cuda_version=cuda_version,
        vram_gb=vram_gb,
        cpu_fallback_reason=reason,
    )


def resolve_torch_device(preference: DevicePreference = "auto") -> torch.device:
    """Return the torch.device used for DINOv2 inference."""
    info = detect_gpu_runtime(preference=preference)
    device = torch.device(info.active_device)
    logger.info(info.summary_for_log())
    return device
