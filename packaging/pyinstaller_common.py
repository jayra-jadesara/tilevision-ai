"""Shared PyInstaller settings — keep Windows, Mac, and Linux builds in sync."""

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
    "watchdog.observers",
    "watchdog.events",
]

EXCLUDES = [
    "matplotlib",
    "notebook",
    "jupyter",
    "torch.distributed",
    "torch.testing",
    "torch.cuda",
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

    resources = project_root / "src" / "resources"
    if resources.is_dir():
        datas.append((str(resources), "src/resources"))

    return datas
