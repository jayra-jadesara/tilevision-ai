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
    Whether to include model_weights/sam2.1-hiera-tiny in the installer.

    TILEVISION_BUNDLE_SAM2:
      0 / false / off  → never (default for production customer builds)
      1 / true / on    → yes when the folder exists
      auto             → Windows + Mac Apple Silicon + Linux; skip Mac Intel
                         (production Intel torch cannot run Transformers SAM2)
    """
    flag = os.environ.get("TILEVISION_BUNDLE_SAM2", "").strip().lower()
    if flag in {"0", "false", "no", "off", ""}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return True
    if flag == "auto":
        arch = (macos_arch or os.environ.get("MACOS_BUILD_ARCH", "")).strip().lower()
        if arch in {"x64", "x86_64", "intel"}:
            return False
        return True
    return False


def sam2_hidden_imports() -> list[str]:
    """Return Sam2 hiddenimports only when the installed transformers has them."""
    if not should_bundle_sam2():
        return []
    try:
        from transformers import Sam2Model  # noqa: F401
    except Exception:
        return []
    return list(_SAM2_HIDDEN_IMPORTS)

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

    # Optional experimental SAM2 Precise Crop weights (~150 MB safetensors).
    if should_bundle_sam2():
        sam2_dir = project_root / "model_weights" / "sam2.1-hiera-tiny"
        if sam2_dir.is_dir() and (sam2_dir / "config.json").is_file():
            datas.append(
                (str(sam2_dir), str(Path("model_weights") / "sam2.1-hiera-tiny"))
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
