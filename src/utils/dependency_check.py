"""
Dependency verification and ordered install steps for first-run setup.

Each step maps import names to pip package names. The setup wizard
installs one step at a time so users without IT support can self-serve.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Sequence

from src.utils.platform_info import (
    cuda_pytorch_install_command,
    detect_display_adapters,
    has_nvidia_gpu,
    is_macos,
)

REQUIRED_STEP_COUNT = 3


@dataclass(frozen=True, slots=True)
class PackageSpec:
    """One dependency checked by the first-run wizard."""

    import_name: str
    pip_name: str
    display_name: str
    builtin: bool = False
    optional: bool = False


@dataclass(frozen=True, slots=True)
class InstallStep:
    """Ordered group shown in the first-run wizard."""

    step_id: str
    title: str
    description: str
    packages: Sequence[PackageSpec]
    optional: bool = False


CURRENT_SETUP_VERSION = 3

INSTALL_STEPS: List[InstallStep] = [
    InstallStep(
        step_id="runtime",
        title=f"Step 1 of {REQUIRED_STEP_COUNT} — Runtime Check",
        description=(
            "Verifies Python 3.12+ and the built-in SQLite engine. "
            "No download is required for this step."
        ),
        packages=(
            PackageSpec("__python__", "", "Python 3.12+", builtin=True),
            PackageSpec("__sqlite__", "", "SQLite Database (built-in)", builtin=True),
        ),
    ),
    InstallStep(
        step_id="core",
        title=f"Step 2 of {REQUIRED_STEP_COUNT} — Application Core",
        description=(
            "Installs the desktop UI, image handling, security, PDF export, "
            "and folder monitoring libraries."
        ),
        packages=(
            PackageSpec("PySide6", "PySide6>=6.6.0", "PySide6"),
            PackageSpec("PIL", "Pillow>=10.0.0", "Pillow"),
            PackageSpec("numpy", "numpy>=1.24.0", "NumPy"),
            PackageSpec("cryptography", "cryptography>=41.0.0", "Cryptography"),
            PackageSpec("reportlab", "reportlab>=4.2.2", "ReportLab"),
            PackageSpec("watchdog", "watchdog>=4.0.0", "Watchdog"),
        ),
    ),
    InstallStep(
        step_id="ai_stack",
        title=f"Step 3 of {REQUIRED_STEP_COUNT} — AI Search Engine",
        description=(
            "Installs PyTorch, DINOv2 embeddings, FAISS vector search, "
            "and image analysis libraries. This is the largest download."
        ),
        packages=(
            PackageSpec("torch", "torch>=2.1.0", "PyTorch"),
            PackageSpec("torchvision", "torchvision>=0.16.0", "torchvision"),
            PackageSpec("transformers", "transformers>=4.45.0", "Transformers"),
            PackageSpec("tokenizers", "tokenizers>=0.19.0", "Tokenizers"),
            PackageSpec("timm", "timm>=1.0.7", "timm"),
            PackageSpec("huggingface_hub", "huggingface-hub>=0.24.0", "Hugging Face Hub"),
            PackageSpec("safetensors", "safetensors>=0.4.3", "safetensors"),
            PackageSpec("faiss", "faiss-cpu>=1.7.4", "FAISS"),
            PackageSpec("cv2", "opencv-python-headless>=4.8.0", "OpenCV"),
            PackageSpec("skimage", "scikit-image>=0.24.0", "scikit-image"),
        ),
    ),
    InstallStep(
        step_id="gpu_optional",
        title="Optional — GPU Acceleration",
        description=(
            "Install CUDA PyTorch when an NVIDIA GPU is present. "
            "Apple Silicon uses MPS automatically; AMD/Intel systems stay on CPU."
        ),
        packages=(
            PackageSpec("__cuda_torch__", "", "CUDA PyTorch (NVIDIA only)", builtin=True),
        ),
        optional=True,
    ),
]


@dataclass(frozen=True, slots=True)
class PackageStatus:
    spec: PackageSpec
    installed: bool
    version: str
    note: str = ""


@dataclass(frozen=True, slots=True)
class StepStatus:
    step: InstallStep
    packages: List[PackageStatus]

    @property
    def is_complete(self) -> bool:
        return step_is_complete(self)


def _package_version(import_name: str) -> str:
    try:
        module = importlib.import_module(import_name)
        return str(getattr(module, "__version__", "installed"))
    except Exception:
        return ""


def _detect_windows_graphics() -> list[str]:
    return detect_display_adapters() if sys.platform == "win32" else []


def is_mps_torch_active() -> bool:
    try:
        import torch
    except Exception:
        return False
    mps = getattr(torch.backends, "mps", None)
    return bool(mps and mps.is_available())


def is_cuda_torch_active() -> bool:
    """Return True when PyTorch can use CUDA."""
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def _check_python_runtime() -> PackageStatus:
    spec = PackageSpec("__python__", "", "Python 3.12+", builtin=True)
    version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    installed = sys.version_info >= (3, 12)
    note = "" if installed else "Python 3.12 or newer is required."
    return PackageStatus(spec=spec, installed=installed, version=version, note=note)


def _check_sqlite_runtime() -> PackageStatus:
    spec = PackageSpec("__sqlite__", "", "SQLite Database (built-in)", builtin=True)
    try:
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE IF NOT EXISTS _tv_probe (id INTEGER PRIMARY KEY)")
        conn.close()
        return PackageStatus(
            spec=spec,
            installed=True,
            version=sqlite3.sqlite_version,
            note="Built into Python — stores tile catalogue metadata.",
        )
    except Exception as exc:
        return PackageStatus(
            spec=spec,
            installed=False,
            version="",
            note=f"SQLite check failed: {exc}",
        )


def _check_cuda_torch_runtime() -> PackageStatus:
    spec = PackageSpec("__cuda_torch__", "", "CUDA PyTorch (NVIDIA only)", builtin=True)

    if is_macos():
        if is_mps_torch_active():
            return PackageStatus(
                spec=spec,
                installed=True,
                version="MPS",
                note="Apple GPU (MPS) is available with the standard PyTorch wheel.",
            )
        return PackageStatus(
            spec=spec,
            installed=True,
            version="CPU",
            note="Intel Mac uses CPU inference — Apple Silicon enables MPS automatically.",
        )

    if not has_nvidia_gpu():
        adapters = detect_display_adapters()
        names = ", ".join(adapters[:2]) if adapters else "integrated graphics"
        return PackageStatus(
            spec=spec,
            installed=True,
            version="not applicable",
            note=f"No NVIDIA GPU ({names}) — CPU mode is used automatically.",
        )

    if is_cuda_torch_active():
        import torch

        device_name = torch.cuda.get_device_name(0)
        return PackageStatus(
            spec=spec,
            installed=True,
            version=torch.__version__,
            note=f"CUDA active on {device_name}.",
        )

    try:
        import torch
    except Exception:
        return PackageStatus(
            spec=spec,
            installed=False,
            version="",
            note="Install CUDA PyTorch to accelerate indexing and search.",
        )

    version = torch.__version__
    if "+cpu" in version.lower():
        note = (
            f"CPU-only PyTorch installed — run: {cuda_pytorch_install_command()}"
        )
    else:
        note = "CUDA PyTorch installed but GPU not active — check NVIDIA driver."
    return PackageStatus(spec=spec, installed=False, version=version, note=note)


def check_package(spec: PackageSpec) -> PackageStatus:
    if spec.import_name == "__python__":
        return _check_python_runtime()
    if spec.import_name == "__sqlite__":
        return _check_sqlite_runtime()
    if spec.import_name == "__cuda_torch__":
        return _check_cuda_torch_runtime()

    try:
        importlib.import_module(spec.import_name)
        return PackageStatus(
            spec=spec,
            installed=True,
            version=_package_version(spec.import_name),
        )
    except Exception:
        return PackageStatus(spec=spec, installed=False, version="")


def step_is_complete(status: StepStatus) -> bool:
    if status.step.optional and status.step.step_id == "gpu_optional":
        return all(pkg.installed for pkg in status.packages)

    for pkg in status.packages:
        if pkg.spec.optional:
            continue
        if not pkg.installed:
            return False
    return True


def check_step(step: InstallStep) -> StepStatus:
    return StepStatus(
        step=step,
        packages=[check_package(spec) for spec in step.packages],
    )


def check_all_steps() -> List[StepStatus]:
    return [check_step(step) for step in INSTALL_STEPS]


def all_dependencies_satisfied() -> bool:
    for status in check_all_steps():
        if status.step.optional:
            continue
        if not step_is_complete(status):
            return False
    return True


def pip_install_packages(packages: Sequence[PackageSpec]) -> tuple[bool, str]:
    """Install missing pip packages for one wizard step."""
    pip_packages = [pkg for pkg in packages if not pkg.builtin and not pkg.optional]
    missing = [pkg for pkg in pip_packages if not check_package(pkg).installed]
    if not missing:
        return True, "Already installed."

    args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        *[spec.pip_name for spec in missing],
    ]
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "Install timed out after 30 minutes."
    except Exception as exc:
        return False, str(exc)

    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-8:])
        return False, tail or f"pip exited with code {completed.returncode}"

    still_missing = [spec.display_name for spec in missing if not check_package(spec).installed]
    if still_missing:
        return False, f"Install finished but missing: {', '.join(still_missing)}"

    return True, "Installed successfully."


def pip_install_cuda_torch() -> tuple[bool, str]:
    """Replace CPU-only PyTorch with the CUDA wheel on NVIDIA PCs."""
    if is_macos():
        if is_mps_torch_active():
            return True, "Apple GPU (MPS) is already active with the standard PyTorch wheel."
        return True, "Intel Mac uses CPU inference. Apple Silicon enables MPS automatically."

    if not has_nvidia_gpu():
        return True, "No NVIDIA GPU detected — CPU mode remains active."

    uninstall = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )

    output = (uninstall.stdout or "") + (uninstall.stderr or "")
    output += (install.stdout or "") + (install.stderr or "")
    if install.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-10:])
        return False, tail or f"pip exited with code {install.returncode}"

    if is_cuda_torch_active():
        import torch

        return True, f"CUDA PyTorch active on {torch.cuda.get_device_name(0)}."

    return False, (
        "CUDA PyTorch installed but GPU is still inactive. "
        "Install the latest NVIDIA driver, then restart TileVision AI."
    )


def install_step_packages(step: InstallStep) -> tuple[bool, str]:
    """Install packages for a wizard step, including optional GPU handling."""
    if step.step_id == "gpu_optional":
        return pip_install_cuda_torch()
    return pip_install_packages(step.packages)
