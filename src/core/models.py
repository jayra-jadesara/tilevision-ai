"""
Domain models for TileVision AI.

Defines pure Python dataclasses representing business entities.
No dependencies on framework or database code.
"""
from src.ai.models import TileFeatures

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class TileImage:
    """
    Represents an indexed tile image entity in the catalog.

    Business metadata is stored directly on this model.
    AI-generated features are grouped into TileFeatures.
    """

    # ------------------------------------------------------------------
    # File Information
    # ------------------------------------------------------------------

    file_path: str
    file_name: str
    file_size: int
    dimensions: str

    # ------------------------------------------------------------------
    # Showroom Metadata
    # ------------------------------------------------------------------

    brand: str = "Unknown"
    category: str = "Unknown"
    color: str = "Unknown"
    size: str = "Unknown"
    product_code: str = "Unknown"

    # ------------------------------------------------------------------
    # Image Metadata
    # ------------------------------------------------------------------

    width: int = 0
    height: int = 0

    sha256_hash: str = ""
    perceptual_hash: str = ""

    # Temporary (will be removed after full migration)
    embedding_id: Optional[int] = None
    # ------------------------------------------------------------------
    # AI Features
    # ------------------------------------------------------------------

    features: Optional[TileFeatures] = None

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    id: Optional[int] = None

    created_time: Optional[datetime] = None

    updated_time: Optional[datetime] = None

    is_indexed: bool = False

    
@dataclass
class LicenseInfo:
    """
    Represents details of the current showroom license installation.
    """
    license_key: str
    hardware_hash: str
    customer_name: Optional[str] = None
    expires_at: Optional[str] = None
    activated_date: Optional[datetime] = None
    id: Optional[int] = None


@dataclass
class SearchResult:
    """
    Represents a matching visual similarity result.
    """
    tile: TileImage
    similarity_score: float
    thumbnail_path: str


@dataclass
class IndexedFolderState:
    """
    Persisted record of a folder that has been indexed at least once.

    Backs Task 1 (Persistent Indexed Folder): lets the Index page restore
    "Folder: X, Indexed Images: N, Status: Ready, Last Indexed: ..." on
    app startup without requiring the user to re-select and re-scan a
    folder they already indexed in a previous session.
    """
    folder_path: str
    last_indexed_at: Optional[datetime] = None
    id: Optional[int] = None
    # Live count of currently-indexed tiles under this folder, populated by
    # the repository/use case query rather than cached in this table — a
    # cached count would drift out of sync if tiles are deleted/added by
    # other means (e.g. auto folder monitoring) between explicit scans.
    indexed_image_count: int = 0


@dataclass
class ScanResult:
    """
    Structured outcome of a folder scan (Task 2: Smart Re-index).

    Distinguishes new vs. modified vs. deleted vs. unchanged files, rather
    than the previous flat (indexed_count, skipped_count) pair, so the UI
    can show a real "what changed" summary instead of just a total.
    """
    new_count: int = 0
    modified_count: int = 0
    deleted_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    total_files_scanned: int = 0
    is_completed: bool = True
    elapsed_seconds: float = 0.0
    time_saved_seconds: float = 0.0

    @property
    def indexed_count(self) -> int:
        """Total files actually (re-)embedded this scan (new + modified)."""
        return self.new_count + self.modified_count

    @property
    def has_any_changes(self) -> bool:
        """True if this scan found anything to do at all."""
        return self.new_count > 0 or self.modified_count > 0 or self.deleted_count > 0


@dataclass
class SearchHistoryEntry:
    """
    A single past search (Task A: Dashboard 'Recent Searches' /
    Task C: Search UX 'search history' — clicking an entry re-runs it).
    """
    query_image_path: str
    result_count: int = 0
    elapsed_seconds: Optional[float] = None
    query_thumbnail_path: Optional[str] = None
    id: Optional[int] = None
    searched_at: Optional[datetime] = None


@dataclass
class ActivityLogEntry:
    """
    A single recent-activity event (Task A: Dashboard 'Recent Activity'),
    e.g. "Indexed 'E:\\Tiles' — 42 new, 3 modified" or "Searched with
    query.jpg — 18 results".
    """
    activity_type: str
    message: str
    id: Optional[int] = None
    created_at: Optional[datetime] = None
