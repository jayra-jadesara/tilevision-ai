"""
Repository interface definitions for TileVision AI.

Declares the abstract base classes for data access components, ensuring
decoupling of the business logic from data storage implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from src.ai.feature_versions import FeatureVersionStatus
from src.core.models import TileImage, LicenseInfo, IndexedFolderState, SearchHistoryEntry, ActivityLogEntry


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
    def get_distinct_values(self, field: str) -> List[str]:
        """
        Retrieve the sorted set of distinct non-empty values present for a
        given metadata column across all indexed tiles — used to populate
        filter dropdowns (Brand/Category/Color/Size) in the Search view.

        Args:
            field: Column name. Must be one of "brand", "category",
                "color", "size" (validated against an allow-list to avoid
                any possibility of SQL injection via this parameter).

        Returns:
            Sorted list of distinct non-empty values for that field.

        Raises:
            ValueError: If field is not an allowed column name.
        """
        pass

    @abstractmethod
    def get_ids_matching_filters(self, filters: Dict[str, str]) -> List[int]:
        """
        Return indexed tile IDs whose metadata matches all provided filters.

        Args:
            filters: Dict of field -> required value (brand/category/color/size).

        Returns:
            Matching tile primary keys.
        """
        pass

    @abstractmethod
    def get_feature_version_status(self) -> FeatureVersionStatus:
        """
        Check whether indexed tiles use the current feature pipeline versions.

        Returns:
            FeatureVersionStatus with compatibility flag and stale counts.
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


class IIndexedFolderRepository(ABC):
    """
    Abstract interface for tracking which folders have been indexed
    (Task 1: Persistent Indexed Folder / Task 2: Smart Re-index).
    """

    @abstractmethod
    def record_folder_indexed(self, folder_path: str) -> None:
        """
        Record (insert or update) that a folder was just (re-)indexed,
        stamping last_indexed_at with the current time.

        Args:
            folder_path: Absolute path of the folder that was scanned.
        """
        pass

    @abstractmethod
    def get_last_indexed_folder(self) -> Optional[IndexedFolderState]:
        """
        Retrieve the most recently indexed folder, for restoring the Index
        page's state on application startup.

        Returns:
            An IndexedFolderState (with indexed_image_count left at its
            default — callers should hydrate that separately via the tile
            repository's live count) if any folder has ever been indexed,
            otherwise None.
        """
        pass

    @abstractmethod
    def get_folder_state(self, folder_path: str) -> Optional[IndexedFolderState]:
        """
        Retrieve the indexed-folder record for a specific folder path.

        Args:
            folder_path: Absolute folder path.

        Returns:
            An IndexedFolderState if this folder has been indexed before,
            otherwise None.
        """
        pass

    @abstractmethod
    def get_all_folders(self) -> List[IndexedFolderState]:
        """
        Retrieve every folder that has ever been indexed (Task A: Dashboard
        'Indexed Folders' stat card).

        Returns:
            A list of IndexedFolderState, most recently indexed first.
        """
        pass


class ISearchHistoryRepository(ABC):
    """
    Abstract interface for recording and retrieving past searches
    (Task A: Dashboard 'Recent Searches' / Task C: Search UX history).
    """

    @abstractmethod
    def record_search(
        self, query_image_path: str, result_count: int,
        elapsed_seconds: Optional[float] = None, query_thumbnail_path: Optional[str] = None,
    ) -> None:
        """
        Record that a search was performed.

        Args:
            query_image_path: Absolute path to the query image used.
            result_count: Number of results the search returned.
            elapsed_seconds: How long the search took, if known.
            query_thumbnail_path: Optional cached thumbnail path for the
                query image, so search history can show a small preview
                without re-reading the (possibly large/temporary) original.
        """
        pass

    @abstractmethod
    def get_recent_searches(self, limit: int = 10) -> List[SearchHistoryEntry]:
        """
        Retrieve the most recent searches, newest first.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            A list of SearchHistoryEntry, newest first.
        """
        pass

    @abstractmethod
    def get_last_search(self) -> Optional[SearchHistoryEntry]:
        """
        Retrieve the single most recent search, for the Dashboard's
        "Last Search" stat card.

        Returns:
            The most recent SearchHistoryEntry, or None if no search has
            ever been performed.
        """
        pass


class IActivityLogRepository(ABC):
    """
    Abstract interface for recording and retrieving recent system activity
    (Task A: Dashboard 'Recent Activity').
    """

    @abstractmethod
    def record_activity(self, activity_type: str, message: str) -> None:
        """
        Record an activity event.

        Args:
            activity_type: Short category tag (e.g. "index", "search",
                "duplicate_scan", "license").
            message: Human-readable description shown on the Dashboard.
        """
        pass

    @abstractmethod
    def get_recent_activity(self, limit: int = 10) -> List[ActivityLogEntry]:
        """
        Retrieve the most recent activity events, newest first.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            A list of ActivityLogEntry, newest first.
        """
        pass
