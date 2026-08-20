"""Shared PyInstaller settings — keep Windows, Mac, and Linux builds in sync."""

from __future__ import annotations

import os
from pathlib import Path

# Packages with lazy imports or hooks that PyInstaller often misses.
HIDDEN_IMPORTS = [
    "transformers",
    "transformers.models.dinov2",
    "timm",
    "safetensors",
    "tokenizers",
    "huggingface_hub",
    "torch",
    "torch.cuda",
    "torch.backends",
    "torch.backends.cudnn",
    "torch.backends.mps",
    "torchvision",
    "faiss",
    "cv2",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "pillow_heif",
    "skimage",
    "cryptography",
    "cryptography.hazmat.primitives.ciphers.aead",
    "cryptography.hazmat.primitives.kdf.pbkdf2",
    "cryptography.hazmat.primitives.asymmetric.ec",
    "cryptography.hazmat.primitives.hashes",
    "certifi",
    "watchdog.observers",
    "watchdog.events",
    "reportlab",
    "reportlab.pdfgen",
    "reportlab.pdfgen.canvas",
    "reportlab.lib.pagesizes",
    "reportlab.lib.utils",
    "onnxruntime",
    "onnxruntime.capi",
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtNetwork",
    "PySide6.QtPrintSupport",
    "PySide6.QtSvg",
    "PySide6.QtTest",
    "shiboken6",
    # Used by packaged-app release validation (S21 memory stress / collectors).
    "psutil",
]

# Optional SAM2 Precise Crop (lab). Only added when TILEVISION_BUNDLE_SAM2 is on
# and transformers exports Sam2Model — never fail a production DINOv2-only build.
_SAM2_HIDDEN_IMPORTS = [
    "transformers.models.sam2",
    "transformers.models.sam2.modeling_sam2",
    "transformers.models.sam2.processing_sam2",
]


def should_bundle_sam2(*, macos_arch: str | None = None) -> bool:
    """
    Whether to include SAM2 ONNX assets in the installer.

    TILEVISION_BUNDLE_SAM2:
      0 / false / off  → never (default for production customer builds)
      1 / true / on / auto → yes on Windows, Mac Intel, Mac Apple Silicon, Linux
                             (identical ONNX package on every platform)
    """
    flag = os.environ.get("TILEVISION_BUNDLE_SAM2", "").strip().lower()
    if flag in {"0", "false", "no", "off", ""}:
        return False
    if flag in {"1", "true", "yes", "on", "auto"}:
        return True
    return False


def should_bundle_sam2_transformers(*, macos_arch: str | None = None) -> bool:
    """
    Optional Transformers safetensors — off by default so Mac/Windows match.

    Enable only with TILEVISION_BUNDLE_SAM2_TRANSFORMERS=1 (lab).
    """
    if not should_bundle_sam2(macos_arch=macos_arch):
        return False
    flag = os.environ.get("TILEVISION_BUNDLE_SAM2_TRANSFORMERS", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def should_bundle_sam2_onnx(*, macos_arch: str | None = None) -> bool:
    """ONNX encoder/decoder — primary Precise Crop path on every OS."""
    return should_bundle_sam2(macos_arch=macos_arch)


def sam2_hidden_imports() -> list[str]:
    """Optional SAM2-related hiddenimports for frozen builds."""
    imports: list[str] = []
    if should_bundle_sam2_onnx():
        imports.extend(
            [
                "onnxruntime",
                "onnxruntime.capi",
                "onnxruntime.capi.onnxruntime_pybind11_state",
            ]
        )
    if should_bundle_sam2_transformers():
        try:
            from transformers import Sam2Model  # noqa: F401
        except Exception:
            pass
        else:
            imports.extend(_SAM2_HIDDEN_IMPORTS)
    return imports

# Never exclude torch.cuda — PyTorch imports it at startup on every platform.
EXCLUDES = [
    "matplotlib",
    "notebook",
    "jupyter",
    "torch.distributed",
    "torch.testing",
    "tensorboard",
    "triton",
    "IPython",
    # Release-validation drivers must load from the CI checkout at runtime —
    # never ship them inside the customer .app (also avoids Analysis pulling
    # the suite in via main.py --release-validation).
    "qa_e2e",
    "pytest",
    "pytest_qt",
    "_pytest",
]


def collect_datas(project_root: Path) -> list[tuple[str, str]]:
    """Data files bundled into every platform build."""
    datas: list[tuple[str, str]] = []

    default_cfg = project_root / "src" / "config" / "default_config.json"
    if default_cfg.is_file():
        datas.append((str(default_cfg), "src/config"))

    model_dir = project_root / "model_weights" / "dinov2-large"
    if model_dir.is_dir():
        datas.append((str(model_dir), str(Path("model_weights") / "dinov2-large")))

    # Optional experimental SAM2 Precise Crop weights.
    if should_bundle_sam2_transformers():
        sam2_dir = project_root / "model_weights" / "sam2.1-hiera-tiny"
        if sam2_dir.is_dir() and (sam2_dir / "config.json").is_file():
            datas.append(
                (str(sam2_dir), str(Path("model_weights") / "sam2.1-hiera-tiny"))
            )
    if should_bundle_sam2_onnx():
        onnx_dir = project_root / "model_weights" / "sam2.1-hiera-tiny-onnx"
        if onnx_dir.is_dir() and any(onnx_dir.glob("*.encoder.onnx")):
            datas.append(
                (str(onnx_dir), str(Path("model_weights") / "sam2.1-hiera-tiny-onnx"))
            )

    resources = project_root / "src" / "resources"
    if resources.is_dir():
        datas.append((str(resources), "src/resources"))

    try:
        import certifi

        datas.append((certifi.where(), "certifi"))
    except Exception:
        pass

    return datas


def collect_hidden_imports() -> list[str]:
    """Base + optional SAM2 hiddenimports for PyInstaller specs."""
    return list(HIDDEN_IMPORTS) + sam2_hidden_imports()


def collect_torch_bundle() -> tuple[list, list, list]:
    """Collect torch + torchvision with all native libs (required for frozen apps)."""
    from PyInstaller.utils.hooks import collect_all

    datas: list = []
    binaries: list = []
    hiddenimports: list = []
    for package in ("torch", "torchvision"):
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    return datas, binaries, hiddenimports


def collect_extra_package_bundles() -> tuple[list, list, list]:
    """
    Collect native-heavy packages PyInstaller often under-bundles on macOS.

    Safe to call when a package is missing (returns empty for that package).
    """
    from PyInstaller.utils.hooks import collect_all

    datas: list = []
    binaries: list = []
    hiddenimports: list = []
    packages = ["onnxruntime", "faiss", "reportlab", "PySide6", "psutil"]
    # NOTE: do NOT collect_all("cv2"). OpenCV's own PyInstaller hooks already
    # collect it; a second collect_all duplicates Frameworks/Resources layout
    # and triggers "recursion is detected during loading of cv2" on macOS .app.
    if should_bundle_sam2_onnx():
        # Already listed; collect_all is idempotent enough for CI.
        pass
    for package in packages:
        try:
            pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
        except Exception:
            continue
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    return datas, binaries, hiddenimports
