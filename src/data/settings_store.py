"""
Settings store module for TileVision AI.

Provides a Clean Architecture boundary wrapper around AppSettings to decouple use cases
from configuration details.
"""

from typing import List

from src.config.settings import AppSettings


class SettingsStore:
    """
    Exposes and manages application settings as a data service.
    
    Acts as a repository layer for application configurations.
    """

    def __init__(self, settings: AppSettings) -> None:
        """
        Initialize the settings store.

        Args:
            settings: An initialized AppSettings instance.
        """
        self._settings = settings

    def get_model_name(self) -> str:
        """Get the current OpenCLIP model name."""
        return self._settings.model_name

    def set_model_name(self, value: str) -> None:
        """Set the OpenCLIP model name."""
        self._settings.model_name = value

    def get_pretrained(self) -> str:
        """Get the current pretrained weights name."""
        return self._settings.pretrained

    def set_pretrained(self, value: str) -> None:
        """Set the pretrained weights name."""
        self._settings.pretrained = value

    def get_database_path(self) -> str:
        """Get the database path."""
        return self._settings.database_path

    def get_index_path(self) -> str:
        """Get the vector index file path."""
        return self._settings.index_path

    def get_thumbnail_dir(self) -> str:
        """Get the cached thumbnails folder path."""
        return self._settings.thumbnail_dir

    def get_watch_folders(self) -> List[str]:
        """Get the list of directories configured for automatic indexing."""
        return self._settings.watch_folders

    def set_watch_folders(self, folders: List[str]) -> None:
        """Set the list of watched directories."""
        self._settings.watch_folders = folders

    def get_top_k(self) -> int:
        """Get default number of similar matches to fetch."""
        return self._settings.top_k

    def set_top_k(self, value: int) -> None:
        """Set the default search limit."""
        self._settings.top_k = value

    def get_theme(self) -> str:
        """Get current interface theme (dark / light)."""
        return self._settings.theme

    def set_theme(self, value: str) -> None:
        """Set user interface theme."""
        self._settings.theme = value

    def get_license_key(self) -> str:
        """Get the saved offline license key."""
        return self._settings.license_key

    def set_license_key(self, value: str) -> None:
        """Save the license key."""
        self._settings.license_key = value
