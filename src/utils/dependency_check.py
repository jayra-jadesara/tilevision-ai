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


@dataclass(frozen=True, slots=True)
class PackageSpec:
    """One pip-installable dependency."""

    import_name: str
    pip_name: str
    display_name: str


@dataclass(frozen=True, slots=True)
class InstallStep:
    """Ordered group shown in the first-run wizard."""

    step_id: str
    title: str
    description: str
    packages: Sequence[PackageSpec]


CURRENT_SETUP_VERSION = 1

INSTALL_STEPS: List[InstallStep] = [
    InstallStep(
        step_id="foundation",
        title="Step 1 of 5 — Application Core",
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
        title="Step 2 of 5 — Tile Image Processing",
        description="OpenCV and scikit-image for catalogue photo analysis.",
        packages=(
            PackageSpec("cv2", "opencv-python-headless>=4.8.0", "OpenCV"),
            PackageSpec("skimage", "scikit-image>=0.24.0", "scikit-image"),
        ),
    ),
    InstallStep(
        step_id="ai_core",
        title="Step 3 of 5 — AI Engine (PyTorch)",
        description="PyTorch and Hugging Face stack for DINOv2 tile embeddings.",
        packages=(
            PackageSpec("torch", "torch>=2.1.0", "PyTorch"),
            PackageSpec("torchvision", "torchvision>=0.16.0", "torchvision"),
            PackageSpec("transformers", "transformers>=4.45.0", "Transformers"),
            PackageSpec("timm", "timm>=1.0.7", "timm"),
            PackageSpec("huggingface_hub", "huggingface-hub>=0.24.0", "Hugging Face Hub"),
            PackageSpec("safetensors", "safetensors>=0.4.3", "safetensors"),
        ),
    ),
    InstallStep(
        step_id="search",
        title="Step 4 of 5 — Visual Search Index",
        description="FAISS vector index for fast tile similarity search.",
        packages=(
            PackageSpec("faiss", "faiss-cpu>=1.7.4", "FAISS (CPU)"),
        ),
    ),
    InstallStep(
        step_id="extras",
        title="Step 5 of 5 — Export and Monitoring",
        description="PDF catalogue export and optional folder auto-indexing.",
        packages=(
            PackageSpec("reportlab", "reportlab>=4.2.2", "ReportLab (PDF)"),
            PackageSpec("watchdog", "watchdog>=4.0.0", "Watchdog (folder monitor)"),
        ),
    ),
]


@dataclass(frozen=True, slots=True)
class PackageStatus:
    spec: PackageSpec
    installed: bool
    version: str


@dataclass(frozen=True, slots=True)
class StepStatus:
    step: InstallStep
    packages: List[PackageStatus]

    @property
    def is_complete(self) -> bool:
        return all(pkg.installed for pkg in self.packages)


def _package_version(import_name: str) -> str:
    try:
        module = importlib.import_module(import_name)
        return str(getattr(module, "__version__", "installed"))
    except Exception:
        return ""


def check_package(spec: PackageSpec) -> PackageStatus:
    try:
        importlib.import_module(spec.import_name)
        return PackageStatus(spec=spec, installed=True, version=_package_version(spec.import_name))
    except Exception:
        return PackageStatus(spec=spec, installed=False, version="")


def check_step(step: InstallStep) -> StepStatus:
    return StepStatus(
        step=step,
        packages=[check_package(spec) for spec in step.packages],
    )


def check_all_steps() -> List[StepStatus]:
    return [check_step(step) for step in INSTALL_STEPS]


OPTIONAL_IMPORTS = frozenset({"watchdog"})


def all_dependencies_satisfied() -> bool:
    for status in check_all_steps():
        for pkg in status.packages:
            if pkg.spec.import_name in OPTIONAL_IMPORTS:
                continue
            if not pkg.installed:
                return False
    return True


def pip_install_packages(packages: Sequence[PackageSpec]) -> tuple[bool, str]:
    """Install missing packages for one wizard step."""
    missing = [pkg for pkg in packages if not check_package(pkg).installed]
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
