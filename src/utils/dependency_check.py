"""
Dependency verification and ordered install steps for first-run setup.

Each step maps import names to pip package names. The setup wizard
installs one step at a time so users without IT support can self-serve.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Sequence

REQUIRED_STEP_COUNT = 6


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


CURRENT_SETUP_VERSION = 2

INSTALL_STEPS: List[InstallStep] = [
    InstallStep(
        step_id="runtime",
        title=f"Step 1 of {REQUIRED_STEP_COUNT} — Runtime and Database",
        description=(
            "Verifies Python and the built-in SQLite database engine. "
            "SQLite ships with Python — no separate download is required."
        ),
        packages=(
            PackageSpec("__python__", "", "Python 3.12+", builtin=True),
            PackageSpec("__sqlite__", "", "SQLite Database (built-in)", builtin=True),
        ),
    ),
    InstallStep(
        step_id="foundation",
        title=f"Step 2 of {REQUIRED_STEP_COUNT} — Application Core",
        description="User interface and security libraries required to run TileVision AI.",
        packages=(
            PackageSpec("PySide6", "PySide6>=6.6.0", "PySide6 (UI)"),
            PackageSpec("PIL", "Pillow>=10.0.0", "Pillow (images)"),
            PackageSpec("numpy", "numpy>=1.24.0", "NumPy"),
            PackageSpec("cryptography", "cryptography>=41.0.0", "Cryptography"),
        ),
    ),
    InstallStep(
        step_id="vision",
        title=f"Step 3 of {REQUIRED_STEP_COUNT} — Tile Image Processing",
        description="OpenCV and scikit-image for catalogue photo analysis.",
        packages=(
            PackageSpec("cv2", "opencv-python-headless>=4.8.0", "OpenCV"),
            PackageSpec("skimage", "scikit-image>=0.24.0", "scikit-image"),
        ),
    ),
    InstallStep(
        step_id="ai_core",
        title=f"Step 4 of {REQUIRED_STEP_COUNT} — AI Engine (PyTorch)",
        description="PyTorch and Hugging Face stack for DINOv2 tile embeddings.",
        packages=(
            PackageSpec("torch", "torch>=2.1.0", "PyTorch (CPU)"),
            PackageSpec("torchvision", "torchvision>=0.16.0", "torchvision"),
            PackageSpec("transformers", "transformers>=4.45.0", "Transformers"),
            PackageSpec("tokenizers", "tokenizers>=0.19.0", "Tokenizers"),
            PackageSpec("timm", "timm>=1.0.7", "timm"),
            PackageSpec("huggingface_hub", "huggingface-hub>=0.24.0", "Hugging Face Hub"),
            PackageSpec("safetensors", "safetensors>=0.4.3", "safetensors"),
        ),
    ),
    InstallStep(
        step_id="search",
        title=f"Step 5 of {REQUIRED_STEP_COUNT} — Visual Search Index",
        description="FAISS vector index for fast tile similarity search.",
        packages=(
            PackageSpec("faiss", "faiss-cpu>=1.7.4", "FAISS (CPU)"),
        ),
    ),
    InstallStep(
        step_id="extras",
        title=f"Step 6 of {REQUIRED_STEP_COUNT} — Export and Monitoring",
        description="PDF catalogue export and optional folder auto-indexing.",
        packages=(
            PackageSpec("reportlab", "reportlab>=4.2.2", "ReportLab (PDF)"),
            PackageSpec("watchdog", "watchdog>=4.0.0", "Watchdog (folder monitor)", optional=True),
        ),
    ),
    InstallStep(
        step_id="gpu_optional",
        title="Optional — NVIDIA GPU Acceleration",
        description=(
            "Install CUDA-enabled PyTorch when an NVIDIA GPU and driver are present. "
            "AMD and Intel graphics use CPU mode automatically — this step is skipped."
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
    if sys.platform != "win32":
        return []

    try:
        completed = subprocess.run(
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
        if completed.returncode != 0:
            return []
        return [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def has_nvidia_gpu() -> bool:
    """Return True when an NVIDIA GPU driver appears to be available."""
    if shutil.which("nvidia-smi"):
        return True
    adapters = _detect_windows_graphics()
    return any("nvidia" in name.lower() for name in adapters)


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
    if not has_nvidia_gpu():
        adapters = _detect_windows_graphics()
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
        note = "CPU-only PyTorch installed — click Install to enable NVIDIA CUDA."
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
