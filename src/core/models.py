"""
Domain models for TileVision AI.

Defines pure Python dataclasses representing business entities.
No dependencies on framework or database code.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class TileImage:
    """
    Represents an indexed tile image entity in the catalog.
    """
    file_path: str
    file_name: str
    file_size: int
    dimensions: str
    id: Optional[int] = None
    date_added: Optional[datetime] = None
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
