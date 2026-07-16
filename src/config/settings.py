"""
Settings management module for TileVision AI.

Handles loading, storing, and modifying application configuration files in a clean,
type-safe manner.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class AppSettings:
    """
    Manages persistent configuration settings for TileVision AI.
    
    Reads from and writes to a JSON configuration file in the user's local AppData directory.
    """

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        """
        Initialize the AppSettings with a specific configuration folder.
        
        Args:
            config_dir: Optional path to the configuration directory. Defaults to ~/.tilevision_ai.
        """
        if config_dir is None:
            self._config_dir = Path.home() / ".tilevision_ai"
        else:
            self._config_dir = config_dir

        self._config_file = self._config_dir / "config.json"
        
        # Ensure directories exist
        self._config_dir.mkdir(parents=True, exist_ok=True)
        
        # Default settings dictionary
        self._defaults: Dict[str, Any] = {
            "model_name": "ViT-B-32-quickgelu",
            "pretrained": "laion400m_e32",
            "database_path": str(self._config_dir / "database" / "tiles.db"),
            "index_path": str(self._config_dir / "index" / "tiles.index"),
            "thumbnail_dir": str(self._config_dir / "thumbnails"),
            "watch_folders": [],
            "top_k": 10,
            "theme": "light",
            "thumbnail_size": 200,
            "license_key": "",
            # Indexing performance (large catalogs: 70–200 MB, 2000+ files)
            "index_batch_size": 12,
            "index_checkpoint_interval": 25,
            "max_decode_edge": 2048,
            "preprocess_workers": 4,
            "large_file_mb_threshold": 10,
            "huge_file_mb_threshold": 50,
            "inference_device": "auto",
            "gpu_index_batch_size": 24,
            "setup_wizard_completed": False,
            "setup_wizard_version": 0,
        }
        
        self._settings: Dict[str, Any] = self._defaults.copy()
        self.load()

    def load(self) -> None:
        """Load settings from the JSON configuration file."""
        if not self._config_file.exists():
            self.save()
            return

        try:
            with open(self._config_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                # Merge loaded configuration with defaults to ensure missing keys are present
                for key, val in self._defaults.items():
                    self._settings[key] = loaded.get(key, val)
        except (json.JSONDecodeError, OSError) as e:
            # Logger is set up after settings, so fallback to print / basic config logging
            print(f"Error loading configuration file '{self._config_file}': {e}. Resetting to defaults.")
            self._settings = self._defaults.copy()
            self.save()

    def save(self) -> None:
        """Save settings to the JSON configuration file."""
        try:
            # Ensure subdirectories for database, index, and thumbnails are initialized
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.thumbnail_dir).mkdir(parents=True, exist_ok=True)

            with open(self._config_file, "w", encoding="utf-8") as f:
                json.dump(self._settings, f, indent=4)
        except OSError as e:
            print(f"Error saving configuration file '{self._config_file}': {e}")

    @property
    def model_name(self) -> str:
        """Get the CLIP model name."""
        return str(self._settings["model_name"])

    @model_name.setter
    def model_name(self, value: str) -> None:
        self._settings["model_name"] = value
        self.save()

    @property
    def pretrained(self) -> str:
        """Get the pre-trained weights name."""
        return str(self._settings["pretrained"])

    @pretrained.setter
    def pretrained(self, value: str) -> None:
        self._settings["pretrained"] = value
        self.save()

    @property
    def database_path(self) -> str:
        """Get the absolute path to the SQLite database."""
        return str(self._settings["database_path"])

    @database_path.setter
    def database_path(self, value: str) -> None:
        self._settings["database_path"] = value
        self.save()

    @property
    def index_path(self) -> str:
        """Get the absolute path to the FAISS index file."""
        return str(self._settings["index_path"])

    @index_path.setter
    def index_path(self, value: str) -> None:
        self._settings["index_path"] = value
        self.save()

    @property
    def thumbnail_dir(self) -> str:
        """Get the directory where tile image thumbnails are stored."""
        return str(self._settings["thumbnail_dir"])

    @thumbnail_dir.setter
    def thumbnail_dir(self, value: str) -> None:
        self._settings["thumbnail_dir"] = value
        self.save()

    @property
    def watch_folders(self) -> List[str]:
        """Get the list of directories monitored for new tiles."""
        return list(self._settings["watch_folders"])

    @watch_folders.setter
    def watch_folders(self, folders: List[str]) -> None:
        self._settings["watch_folders"] = [str(Path(f).resolve()) for f in folders]
        self.save()

    @property
    def top_k(self) -> int:
        """Get the default number of search results to return."""
        return int(self._settings["top_k"])

    @top_k.setter
    def top_k(self, value: int) -> None:
        self._settings["top_k"] = value
        self.save()

    @property
    def theme(self) -> str:
        """Get the preferred UI theme name."""
        return str(self._settings["theme"])

    @theme.setter
    def theme(self, value: str) -> None:
        self._settings["theme"] = value
        self.save()

    @property
    def thumbnail_size(self) -> int:
        """Get the preferred thumbnail size in pixels (Task D: Settings)."""
        return int(self._settings.get("thumbnail_size", 200))

    @thumbnail_size.setter
    def thumbnail_size(self, value: int) -> None:
        self._settings["thumbnail_size"] = int(value)
        self.save()

    @property
    def license_key(self) -> str:
        """Get the stored cryptographic offline license key."""
        return str(self._settings["license_key"])

    @license_key.setter
    def license_key(self, value: str) -> None:
        self._settings["license_key"] = value
        self.save()

    @property
    def index_batch_size(self) -> int:
        """Images per DINOv2 batch during folder scans."""
        return int(self._settings.get("index_batch_size", 12))

    @index_batch_size.setter
    def index_batch_size(self, value: int) -> None:
        self._settings["index_batch_size"] = max(1, int(value))

    @property
    def index_checkpoint_interval(self) -> int:
        """Save FAISS index to disk every N processed files."""
        return int(self._settings.get("index_checkpoint_interval", 25))

    @index_checkpoint_interval.setter
    def index_checkpoint_interval(self, value: int) -> None:
        self._settings["index_checkpoint_interval"] = max(1, int(value))

    @property
    def max_decode_edge(self) -> int:
        """Max pixel edge when decoding source images (memory guard)."""
        return int(self._settings.get("max_decode_edge", 2048))

    @max_decode_edge.setter
    def max_decode_edge(self, value: int) -> None:
        self._settings["max_decode_edge"] = max(512, int(value))

    @property
    def preprocess_workers(self) -> int:
        """Parallel workers for image preprocessing during batch indexing."""
        return int(self._settings.get("preprocess_workers", 4))

    @preprocess_workers.setter
    def preprocess_workers(self, value: int) -> None:
        self._settings["preprocess_workers"] = max(1, int(value))

    @property
    def large_file_mb_threshold(self) -> int:
        """Files >= this size use smaller batches and fewer workers."""
        return int(self._settings.get("large_file_mb_threshold", 10))

    @large_file_mb_threshold.setter
    def large_file_mb_threshold(self, value: int) -> None:
        self._settings["large_file_mb_threshold"] = max(1, int(value))

    @property
    def huge_file_mb_threshold(self) -> int:
        """Files >= this size use minimal batch size (2) and 1 worker."""
        return int(self._settings.get("huge_file_mb_threshold", 50))

    @huge_file_mb_threshold.setter
    def huge_file_mb_threshold(self, value: int) -> None:
        self._settings["huge_file_mb_threshold"] = max(1, int(value))

    @property
    def inference_device(self) -> str:
        """DINOv2 device: auto, cuda, or cpu."""
        value = str(self._settings.get("inference_device", "auto")).lower()
        return value if value in ("auto", "cuda", "cpu") else "auto"

    @inference_device.setter
    def inference_device(self, value: str) -> None:
        normalized = str(value).lower()
        self._settings["inference_device"] = (
            normalized if normalized in ("auto", "cuda", "cpu") else "auto"
        )

    @property
    def gpu_index_batch_size(self) -> int:
        """Batch size used when CUDA is active (larger = faster indexing)."""
        return int(self._settings.get("gpu_index_batch_size", 24))

    @gpu_index_batch_size.setter
    def gpu_index_batch_size(self, value: int) -> None:
        self._settings["gpu_index_batch_size"] = max(1, int(value))

    @property
    def setup_wizard_completed(self) -> bool:
        return bool(self._settings.get("setup_wizard_completed", False))

    @setup_wizard_completed.setter
    def setup_wizard_completed(self, value: bool) -> None:
        self._settings["setup_wizard_completed"] = bool(value)
        self.save()

    @property
    def setup_wizard_version(self) -> int:
        return int(self._settings.get("setup_wizard_version", 0))

    @setup_wizard_version.setter
    def setup_wizard_version(self, value: int) -> None:
        self._settings["setup_wizard_version"] = int(value)
        self.save()
