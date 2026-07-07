"""
Repository interface definitions for TileVision AI.

Declares the abstract base classes for data access components, ensuring
decoupling of the business logic from data storage implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.core.models import TileImage, LicenseInfo


class IImageRepository(ABC):
    """Abstract interface for managing tile image metadata storage."""

    @abstractmethod
    def add(self, tile: TileImage) -> int:
        """
        Persist a new tile image metadata record.

        Args:
            tile: The TileImage dataclass model.

        Returns:
            The generated unique ID of the inserted record.
        """
        pass

    @abstractmethod
    def remove(self, image_id: int) -> bool:
        """
        Delete a tile image record by ID.

        Args:
            image_id: Unique database ID of the tile.

        Returns:
            True if deletion was successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_by_id(self, image_id: int) -> Optional[TileImage]:
        """
        Retrieve a tile image record by unique ID.

        Args:
            image_id: Unique database ID.

        Returns:
            The TileImage dataclass if found, otherwise None.
        """
        pass

    @abstractmethod
    def get_by_path(self, file_path: str) -> Optional[TileImage]:
        """
        Retrieve a tile image record by its absolute file path.

        Args:
            file_path: The absolute path string.

        Returns:
            The TileImage dataclass if found, otherwise None.
        """
        pass

    @abstractmethod
    def get_all(self) -> List[TileImage]:
        """
        Retrieve all persisted tile image records.

        Returns:
            A list of all TileImage records.
        """
        pass

    @abstractmethod
    def get_by_ids(self, ids: List[int]) -> List[TileImage]:
        """
        Retrieve a specific list of tile images by their IDs (maintains input order).

        Args:
            ids: List of database integer keys.

        Returns:
            A list of TileImage records corresponding to input IDs.
        """
        pass

    @abstractmethod
    def get_pending_index(self) -> List[TileImage]:
        """
        Retrieve all tile records that have been added to DB but not yet indexed in FAISS.

        Returns:
            A list of TileImage records where is_indexed is False.
        """
        pass

    @abstractmethod
    def mark_as_indexed(self, image_id: int, is_indexed: bool) -> bool:
        """
        Update the FAISS indexing status flag of a tile record.

        Args:
            image_id: Unique database ID.
            is_indexed: Indexing status boolean.

        Returns:
            True if update succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def clear_all(self) -> None:
        """Delete all tile image records from the repository."""
        pass


class ILicenseRepository(ABC):
    """Abstract interface for managing offline licensing key storage."""

    @abstractmethod
    def save_license(self, license_info: LicenseInfo) -> bool:
        """
        Persist/update a license key installation record.

        Args:
            license_info: The LicenseInfo data model.

        Returns:
            True if save was successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_license(self) -> Optional[LicenseInfo]:
        """
        Retrieve the latest installed license key details.

        Returns:
            The LicenseInfo model if installed, otherwise None.
        """
        pass

    @abstractmethod
    def clear_license(self) -> None:
        """Remove all license activation keys from storage."""
        pass
