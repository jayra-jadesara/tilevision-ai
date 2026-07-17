"""
SQLite implementation of the repository interfaces for TileVision AI.

Handles database transactions and data conversions between SQLite Rows
and domain models.
"""

from datetime import datetime
import logging
import sqlite3
from typing import Dict, List, Optional
import numpy as np

from src.core.models import TileImage, LicenseInfo, IndexedFolderState, SearchHistoryEntry, ActivityLogEntry
from src.ai.models import TileFeatures
from src.ai.feature_versions import (
    CURRENT_FEATURE_VERSION,
    CURRENT_PATTERN_FEATURE_VERSION,
    CURRENT_EMBEDDING_MODEL,
    CURRENT_EMBEDDING_DIMENSION,
    CURRENT_COLOR_HISTOGRAM_SIZE,
    FeatureVersionStatus,
    is_tile_features_compatible,
)

from src.data.db_context import DatabaseContext
from src.data.repository_interface import (
    IImageRepository, ILicenseRepository, IIndexedFolderRepository,
    ISearchHistoryRepository, IActivityLogRepository, ICatalogueProfileRepository,
)

logger = logging.getLogger("tilevision.data.sqlite_repository")


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """
    Best-effort parser for SQLite CURRENT_TIMESTAMP / ISO-format strings.
    Shared across all repositories in this module so the same fallback
    parsing logic doesn't get duplicated (and drift out of sync) in each
    _row_to_entity() method.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        pass
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        logger.warning(f"Could not parse timestamp value: {value}")
        return None


class SQLiteImageRepository(IImageRepository):
    """SQLite-backed repository for managing TileImage entities."""

    def __init__(self, db_context: DatabaseContext) -> None:
        """
        Initialize the repository.

        Args:
            db_context: The shared DatabaseContext instance.
        """
        self._db = db_context

    def _row_to_entity(self, row: sqlite3.Row) -> TileImage:
        """Helper to convert a sqlite3.Row to a TileImage model."""

        created_time = _parse_timestamp(row["created_time"])
        updated_time = _parse_timestamp(row["updated_time"])

        # -------------------------------------------------------
        # Reconstruct AI features from SQLite
        # -------------------------------------------------------

        features = None

        has_complete_features = all(
            row[name] is not None
            for name in (
                "embedding_blob",
                "color_histogram",
                "texture_histogram",
                "edge_histogram",
                "pattern_features",
                "dominant_r",
                "dominant_g",
                "dominant_b",
            )
        )

        if has_complete_features:
            features = TileFeatures(
                # DINOv2 embedding is stored as float32
                embedding=self._deserialize_vector(
                    row["embedding_blob"]
                ),

                # Histograms are stored as float16
                color_histogram=self._deserialize_histogram(
                    row["color_histogram"]
                ),

                texture_histogram=self._deserialize_histogram(
                    row["texture_histogram"]
                ),

                edge_histogram=self._deserialize_histogram(
                    row["edge_histogram"]
                ),

                # Pattern features are stored as float32
                pattern_features=self._deserialize_vector(
                    row["pattern_features"]
                ),

                dominant_color=(
                    row["dominant_r"],
                    row["dominant_g"],
                    row["dominant_b"],
                ),

                width=row["width"],
                height=row["height"],
            )

         # -------------------------------------------------------
        # Build TileImage entity
        # -------------------------------------------------------

        return TileImage(
            id=row["id"],
            file_path=row["file_path"],
            file_name=row["file_name"],
            file_size=row["file_size"],
            dimensions=row["dimensions"],
            file_mtime=float(row["file_mtime"] or 0.0),
            brand=row["brand"],
            category=row["category"],
            color=row["color"],
            size=row["size"],
            product_code=row["product_code"],
            width=row["width"],
            height=row["height"],
            sha256_hash=row["sha256_hash"],
            perceptual_hash=row["perceptual_hash"],
            embedding_id=row["embedding_id"],
            features=features,
            created_time=created_time,
            updated_time=updated_time,
            is_indexed=bool(row["is_indexed"]),
        )

    def add(self, tile: TileImage) -> int:
        """
        Add or update a tile image metadata record in SQLite.

        Args:
            tile: The TileImage dataclass.

        Returns:
            The primary key of the inserted/updated record.
        """
        query = """
        INSERT INTO tiles (

            file_path,
            file_name,
            file_size,
            dimensions,
            file_mtime,

            brand,
            category,
            color,
            size,
            product_code,

            width,
            height,

            sha256_hash,
            perceptual_hash,

            embedding_id,

            embedding_blob,
            embedding_dimension,
            embedding_model,

            color_histogram,
            texture_histogram,
            edge_histogram,
            pattern_features,

            dominant_r,
            dominant_g,
            dominant_b,

            feature_version,
            pattern_feature_version,

            is_indexed,

            updated_time
        )
        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?,
            CURRENT_TIMESTAMP
        )
        ON CONFLICT(file_path) DO UPDATE SET
            file_name=excluded.file_name,
            file_size=excluded.file_size,
            dimensions=excluded.dimensions,
            file_mtime=excluded.file_mtime,
            brand=excluded.brand,
            category=excluded.category,
            color=excluded.color,
            size=excluded.size,
            product_code=excluded.product_code,
            width=excluded.width,
            height=excluded.height,
            sha256_hash=excluded.sha256_hash,
            perceptual_hash=excluded.perceptual_hash,
            embedding_id=excluded.embedding_id,
            embedding_blob=excluded.embedding_blob,
            embedding_dimension=excluded.embedding_dimension,
            embedding_model=excluded.embedding_model,
            color_histogram=excluded.color_histogram,
            texture_histogram=excluded.texture_histogram,
            edge_histogram=excluded.edge_histogram,
            pattern_features=excluded.pattern_features,
            dominant_r=excluded.dominant_r,
            dominant_g=excluded.dominant_g,
            dominant_b=excluded.dominant_b,
            feature_version=excluded.feature_version,
            pattern_feature_version=excluded.pattern_feature_version,
            is_indexed=excluded.is_indexed,
            updated_time=CURRENT_TIMESTAMP
        RETURNING id;
        """
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                features = tile.features
                embedding_blob = None

                embedding_dimension = None

                embedding_model = None

                color_blob = None

                texture_blob = None

                edge_blob = None

                pattern_features = None

                r = g = b = 0
                feature_version = 0
                pattern_feature_version = 0

                if features is not None:

                    embedding_blob = self._serialize_vector(
                        features.embedding
                    )

                    embedding_dimension = (
                        len(features.embedding)
                        if features.embedding is not None
                        else None
                    )

                    embedding_model = CURRENT_EMBEDDING_MODEL

                    color_blob = self._serialize_histogram(
                        features.color_histogram
                    )

                    texture_blob = self._serialize_histogram(
                        features.texture_histogram
                    )

                    edge_blob = self._serialize_histogram(
                        features.edge_histogram
                    )

                    pattern_features = self._serialize_vector(
                        features.pattern_features
                    )

                    r = int(features.dominant_color[0])
                    g = int(features.dominant_color[1])
                    b = int(features.dominant_color[2])

                    feature_version = CURRENT_FEATURE_VERSION
                    pattern_feature_version = CURRENT_PATTERN_FEATURE_VERSION
                    
                cursor.execute(
                    query,
                    (
                        tile.file_path,
                        tile.file_name,
                        tile.file_size,
                        tile.dimensions,
                        float(tile.file_mtime or 0.0),

                        tile.brand,
                        tile.category,
                        tile.color,
                        tile.size,
                        tile.product_code,

                        tile.width,
                        tile.height,

                        tile.sha256_hash,
                        tile.perceptual_hash,

                        tile.embedding_id,

                        embedding_blob,
                        embedding_dimension,
                        embedding_model,

                        color_blob,
                        texture_blob,
                        edge_blob,
                        pattern_features,

                        r,
                        g,
                        b,

                        feature_version,
                        pattern_feature_version,

                        int(tile.is_indexed),
                    ),
                )
                row = cursor.fetchone()
                conn.commit()
                if row:
                    generated_id = row["id"]
                    logger.debug(f"Saved tile '{tile.file_name}' with ID: {generated_id}")
                    return int(generated_id)
                else:
                    # Fallback if RETURNING clause isn't supported (older SQLite versions)
                    return int(cursor.lastrowid or 0)
        except sqlite3.Error as e:
            logger.error(f"Failed to add image to repository: {e}")
            raise RuntimeError(f"Database error during add operation: {e}") from e

    def remove(self, image_id: int) -> bool:
        """
        Remove a tile image record by ID.

        Args:
            image_id: Unique database ID.

        Returns:
            True if a row was deleted, False otherwise.
        """
        query = "DELETE FROM tiles WHERE id = ?;"
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (image_id,))
                conn.commit()
                success = cursor.rowcount > 0
                logger.info(f"Removed tile ID {image_id}: {success}")
                return success
        except sqlite3.Error as e:
            logger.error(f"Failed to delete tile ID {image_id}: {e}")
            return False

    def get_by_id(self, image_id: int) -> Optional[TileImage]:
        """
        Fetch a single tile image by ID.

        Args:
            image_id: Unique database ID.

        Returns:
            The TileImage if found, otherwise None.
        """
        query = "SELECT * FROM tiles WHERE id = ?;"
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (image_id,))
                row = cursor.fetchone()
                if row:
                    return self._row_to_entity(row)
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch tile by ID {image_id}: {e}")
        return None

    def get_by_path(self, file_path: str) -> Optional[TileImage]:
        """
        Fetch a single tile image by absolute path.

        Args:
            file_path: The absolute path string.

        Returns:
            The TileImage if found, otherwise None.
        """
        query = "SELECT * FROM tiles WHERE file_path = ?;"
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (file_path,))
                row = cursor.fetchone()
                if row:
                    return self._row_to_entity(row)
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch tile by path {file_path}: {e}")
        return None

    def get_all(self) -> List[TileImage]:
        """
        Retrieve all tile records.

        Returns:
            A list of TileImage models.
        """
        query = "SELECT * FROM tiles ORDER BY created_time DESC;"
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
                return [self._row_to_entity(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch all tiles: {e}")
            return []

    def get_by_ids(self, ids: List[int]) -> List[TileImage]:
        """
        Fetch multiple tile records by their IDs.
        Maintains the exact order of the input ID list (crucial for FAISS similarity sorting).

        Args:
            ids: List of database keys.

        Returns:
            Ordered list of matching TileImage models.
        """
        if not ids:
            return []

        # Prepare placeholder query
        placeholders = ",".join("?" for _ in ids)
        query = f"SELECT * FROM tiles WHERE id IN ({placeholders});"
        
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(query, ids)
                rows = cursor.fetchall()
                
                # Load rows into a dictionary for fast re-ordering lookup
                entities_dict = {}
                for row in rows:
                    entity = self._row_to_entity(row)
                    if entity.id is not None:
                        entities_dict[entity.id] = entity
                
                # Reassemble list matching the exact order of queried IDs
                ordered_results = []
                for image_id in ids:
                    if image_id in entities_dict:
                        ordered_results.append(entities_dict[image_id])
                return ordered_results
        except sqlite3.Error as e:
            logger.error(f"Failed to batch fetch tiles: {e}")
            return []

    def get_pending_index(self) -> List[TileImage]:
        """
        Get all images that are in database but not yet indexed in FAISS.

        Returns:
            A list of non-indexed TileImage models.
        """
        query = "SELECT * FROM tiles WHERE is_indexed = 0;"
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
                return [self._row_to_entity(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch pending index tiles: {e}")
            return []

    @staticmethod
    def _serialize_vector(
        vector: np.ndarray | None,
    ) -> bytes | None:
        """
        Serialize a NumPy feature vector for SQLite BLOB storage.

        Used for:
        - DINOv2 embedding
        - pattern features
        """
        if vector is None:
            return None

        vector = np.asarray(
            vector,
            dtype=np.float32,
        )

        return vector.tobytes()


    @staticmethod
    def _deserialize_vector(
        blob: bytes | None,
    ) -> np.ndarray | None:
        """
        Deserialize SQLite BLOB back into a float32 NumPy vector.
        """
        if blob is None:
            return None

        return np.frombuffer(
            blob,
            dtype=np.float32,
        ).copy()
    
    @staticmethod
    def _serialize_histogram(hist: np.ndarray | None) -> bytes | None:

        if hist is None:
            return None

        return hist.astype(
            np.float16
        ).tobytes()

    @staticmethod
    def _deserialize_histogram(blob: bytes | None) -> np.ndarray | None:

        if blob is None:
            return None

        return np.frombuffer(
            blob,
            dtype=np.float16,
        ).astype(np.float32)



    # Allow-list of columns that may be queried via get_distinct_values().
    # Never interpolate the caller-supplied field name directly into SQL —
    # even though it's an internal API today, validating against a fixed
    # set here means a future caller (e.g. a filter UI wired to raw user
    # input) can't turn this into a SQL injection vector.
    _DISTINCT_VALUE_ALLOWED_FIELDS = frozenset({"brand", "category", "color", "size"})

    def get_distinct_values(self, field: str) -> List[str]:
        """
        Retrieve the sorted set of distinct non-empty values for a metadata
        column, for populating filter dropdowns.

        Args:
            field: One of "brand", "category", "color", "size".

        Returns:
            Sorted list of distinct non-empty values.

        Raises:
            ValueError: If field is not an allowed column name.
        """
        if field not in self._DISTINCT_VALUE_ALLOWED_FIELDS:
            raise ValueError(
                f"Invalid field '{field}' for get_distinct_values(). "
                f"Must be one of: {sorted(self._DISTINCT_VALUE_ALLOWED_FIELDS)}"
            )

        # Safe to f-string the column name here ONLY because it was just
        # validated against the fixed allow-list above, never from raw input.
        query = (
            f"SELECT DISTINCT {field} FROM tiles "
            f"WHERE is_indexed = 1 "
            f"AND {field} IS NOT NULL AND TRIM({field}) != '' "
            f"AND LOWER(TRIM({field})) != 'unknown' "
            f"ORDER BY {field} COLLATE NOCASE;"
        )
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
                return [row[field] for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch distinct values for field '{field}': {e}")
            return []

    def get_ids_matching_filters(self, filters: Dict[str, str]) -> List[int]:
        """
        Return indexed tile IDs matching all active metadata filters.
        """
        if not filters:
            return []

        clauses = ["is_indexed = 1"]
        params: List[str] = []

        for field, value in filters.items():
            if field not in self._DISTINCT_VALUE_ALLOWED_FIELDS:
                continue
            if not value or str(value).strip().lower() == "unknown":
                continue
            clauses.append(f"LOWER(TRIM({field})) = LOWER(TRIM(?))")
            params.append(str(value).strip())

        if len(params) == 0:
            return []

        query = f"SELECT id FROM tiles WHERE {' AND '.join(clauses)} ORDER BY id;"
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [int(row["id"]) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch IDs for filters {filters}: {e}")
            return []

    def get_feature_version_status(self) -> FeatureVersionStatus:
        """
        Check whether indexed tiles use the current feature pipeline versions.
        """
        query = """
        SELECT
            feature_version,
            pattern_feature_version,
            embedding_model,
            embedding_dimension,
            pattern_features,
            color_histogram
        FROM tiles
        WHERE is_indexed = 1;
        """
        indexed_count = 0
        stale_count = 0

        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()

                for row in rows:
                    indexed_count += 1
                    pattern_size = None
                    color_size = None
                    if row["pattern_features"] is not None:
                        pattern_size = (
                            len(row["pattern_features"]) // 4
                        )
                    if row["color_histogram"] is not None:
                        color_size = (
                            len(row["color_histogram"]) // 2
                        )

                    if not is_tile_features_compatible(
                        feature_version=row["feature_version"],
                        pattern_feature_version=row["pattern_feature_version"],
                        embedding_model=row["embedding_model"],
                        embedding_dimension=row["embedding_dimension"],
                        pattern_feature_size=pattern_size,
                        color_histogram_size=color_size,
                    ):
                        stale_count += 1
        except sqlite3.Error as e:
            logger.error(f"Failed to check feature versions: {e}")
            return FeatureVersionStatus(
                is_compatible=False,
                indexed_count=0,
                stale_count=0,
                message="Could not verify feature versions.",
            )

        if indexed_count == 0:
            return FeatureVersionStatus(
                is_compatible=True,
                indexed_count=0,
                stale_count=0,
                message="No indexed tiles yet.",
            )

        is_compatible = stale_count == 0
        if is_compatible:
            message = (
                f"All {indexed_count} indexed tile(s) use current "
                f"feature version {CURRENT_FEATURE_VERSION}."
            )
        else:
            message = (
                f"{stale_count} of {indexed_count} indexed tile(s) use "
                f"stale features. A full re-index is required."
            )

        return FeatureVersionStatus(
            is_compatible=is_compatible,
            indexed_count=indexed_count,
            stale_count=stale_count,
            message=message,
        )

    def mark_as_indexed(self, image_id: int, is_indexed: bool) -> bool:
        """
        Update the is_indexed status for a tile.

        Args:
            image_id: Unique database ID.
            is_indexed: Flag value (True/False).

        Returns:
            True if the record was updated, False otherwise.
        """
        query = "UPDATE tiles SET is_indexed = ? WHERE id = ?;"
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (int(is_indexed), image_id))
                conn.commit()
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Failed to update index status for ID {image_id}: {e}")
            return False

    def clear_all(self) -> None:
        """Truncate the tiles database table."""
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM tiles;")
                conn.commit()
                # Run vacuum to reclaim space in SQLite file
                cursor.execute("VACUUM;")
                conn.commit()
            logger.info("Cleared all tiles records from SQLite database.")
        except sqlite3.Error as e:
            logger.error(f"Failed to clear tiles repository: {e}")
            raise RuntimeError(f"Database clear failure: {e}") from e


class SQLiteLicenseRepository(ILicenseRepository):
    """SQLite-backed repository for managing LicenseInfo entities."""

    def __init__(self, db_context: DatabaseContext) -> None:
        """
        Initialize the repository.

        Args:
            db_context: Shared DatabaseContext.
        """
        self._db = db_context

    def _row_to_entity(self, row: sqlite3.Row) -> LicenseInfo:
        """Helper to convert a sqlite3.Row to a LicenseInfo model."""
        activated_date = _parse_timestamp(row["activated_date"])

        return LicenseInfo(
            id=row["id"],
            license_key=row["license_key"],
            hardware_hash=row["hardware_hash"],
            customer_name=row["customer_name"],
            expires_at=row["expires_at"],
            activated_date=activated_date,
        )

    def save_license(self, license_info: LicenseInfo) -> bool:
        """
        Save or update the license key in SQLite.
        Clears previous installations first.

        Args:
            license_info: The LicenseInfo model.

        Returns:
            True if saving succeeded, False otherwise.
        """
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                # Enforce single active license in database
                cursor.execute("DELETE FROM license_info;")
                
                query = """
                INSERT INTO license_info (license_key, customer_name, expires_at, hardware_hash)
                VALUES (?, ?, ?, ?);
                """
                cursor.execute(
                    query,
                    (
                        license_info.license_key,
                        license_info.customer_name,
                        license_info.expires_at,
                        license_info.hardware_hash,
                    ),
                )
                conn.commit()
                logger.info(f"Saved new license key for {license_info.customer_name}")
                return True
        except sqlite3.Error as e:
            logger.error(f"Failed to save license record: {e}")
            return False

    def get_license(self) -> Optional[LicenseInfo]:
        """
        Retrieve the currently installed license.

        Returns:
            The LicenseInfo model if found, otherwise None.
        """
        query = "SELECT * FROM license_info ORDER BY id DESC LIMIT 1;"
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                row = cursor.fetchone()
                if row:
                    return self._row_to_entity(row)
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve license record: {e}")
        return None

    def clear_license(self) -> None:
        """Remove the installed license key registration."""
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM license_info;")
                conn.commit()
            logger.info("Cleared license record from database.")
        except sqlite3.Error as e:
            logger.error(f"Failed to clear license: {e}")
            raise RuntimeError(f"Database error during license clear: {e}") from e


class SQLiteIndexedFolderRepository(IIndexedFolderRepository):
    """
    SQLite-backed repository for tracking indexed folders (Task 1: Persistent
    Indexed Folder / Task 2: Smart Re-index).
    """

    def __init__(self, db_context: DatabaseContext) -> None:
        """
        Args:
            db_context: Shared DatabaseContext.
        """
        self._db = db_context

    def record_folder_indexed(self, folder_path: str) -> None:
        """
        Insert or update the last_indexed_at timestamp for a folder.

        Args:
            folder_path: Absolute path of the folder that was scanned.
        """
        # NOTE: uses strftime(..., 'now') for millisecond-resolution
        # timestamps rather than CURRENT_TIMESTAMP, which only has
        # second-resolution — two folders indexed within the same second
        # would otherwise tie and sort unpredictably in
        # get_last_indexed_folder()'s ORDER BY.
        query = """
        INSERT INTO indexed_folders (folder_path, last_indexed_at)
        VALUES (?, strftime('%Y-%m-%d %H:%M:%f', 'now'))
        ON CONFLICT(folder_path) DO UPDATE SET
            last_indexed_at = strftime('%Y-%m-%d %H:%M:%f', 'now');
        """
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (folder_path,))
                conn.commit()
            logger.info(f"Recorded folder as indexed: {folder_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to record indexed folder '{folder_path}': {e}")
            raise RuntimeError(f"Database error recording indexed folder: {e}") from e

    def get_last_indexed_folder(self) -> Optional[IndexedFolderState]:
        """
        Retrieve the most recently indexed folder (by last_indexed_at).

        Returns:
            An IndexedFolderState if any folder has been indexed, else None.
        """
        query = "SELECT * FROM indexed_folders ORDER BY last_indexed_at DESC LIMIT 1;"
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                row = cursor.fetchone()
                if row:
                    return self._row_to_entity(row)
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch last indexed folder: {e}")
        return None

    def get_folder_state(self, folder_path: str) -> Optional[IndexedFolderState]:
        """
        Retrieve the indexed-folder record for a specific path.

        Args:
            folder_path: Absolute folder path.

        Returns:
            An IndexedFolderState if found, else None.
        """
        query = "SELECT * FROM indexed_folders WHERE folder_path = ?;"
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (folder_path,))
                row = cursor.fetchone()
                if row:
                    return self._row_to_entity(row)
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch folder state for '{folder_path}': {e}")
        return None

    def get_all_folders(self) -> List[IndexedFolderState]:
        """
        Retrieve every folder that has ever been indexed.

        Returns:
            A list of IndexedFolderState, most recently indexed first.
        """
        query = "SELECT * FROM indexed_folders ORDER BY last_indexed_at DESC;"
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
                return [self._row_to_entity(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch all indexed folders: {e}")
            return []

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> IndexedFolderState:
        return IndexedFolderState(
            id=row["id"],
            folder_path=row["folder_path"],
            last_indexed_at=_parse_timestamp(row["last_indexed_at"]),
        )


class SQLiteSearchHistoryRepository(ISearchHistoryRepository):
    """
    SQLite-backed repository for search history (Task A: Dashboard /
    Task C: Search UX).
    """

    def __init__(self, db_context: DatabaseContext) -> None:
        self._db = db_context

    def record_search(
        self, query_image_path: str, result_count: int,
        elapsed_seconds: Optional[float] = None, query_thumbnail_path: Optional[str] = None,
    ) -> None:
        query = """
        INSERT INTO search_history (query_image_path, query_thumbnail_path, result_count, elapsed_seconds)
        VALUES (?, ?, ?, ?);
        """
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (query_image_path, query_thumbnail_path, result_count, elapsed_seconds))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to record search history: {e}")

    def get_recent_searches(self, limit: int = 10) -> List[SearchHistoryEntry]:
        query = "SELECT * FROM search_history ORDER BY searched_at DESC LIMIT ?;"
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (limit,))
                rows = cursor.fetchall()
                return [self._row_to_entity(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch recent searches: {e}")
            return []

    def get_last_search(self) -> Optional[SearchHistoryEntry]:
        results = self.get_recent_searches(limit=1)
        return results[0] if results else None

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> SearchHistoryEntry:
        return SearchHistoryEntry(
            id=row["id"],
            query_image_path=row["query_image_path"],
            query_thumbnail_path=row["query_thumbnail_path"],
            result_count=row["result_count"],
            elapsed_seconds=row["elapsed_seconds"],
            searched_at=_parse_timestamp(row["searched_at"]),
        )


class SQLiteActivityLogRepository(IActivityLogRepository):
    """SQLite-backed repository for the Dashboard's Recent Activity feed."""

    def __init__(self, db_context: DatabaseContext) -> None:
        self._db = db_context

    def record_activity(self, activity_type: str, message: str) -> None:
        query = "INSERT INTO activity_log (activity_type, message) VALUES (?, ?);"
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (activity_type, message))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to record activity log entry: {e}")

    def get_recent_activity(self, limit: int = 10) -> List[ActivityLogEntry]:
        query = "SELECT * FROM activity_log ORDER BY created_at DESC LIMIT ?;"
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (limit,))
                rows = cursor.fetchall()
                return [
                    ActivityLogEntry(
                        id=row["id"],
                        activity_type=row["activity_type"],
                        message=row["message"],
                        created_at=_parse_timestamp(row["created_at"]),
                    )
                    for row in rows
                ]
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch recent activity: {e}")
            return []


class SQLiteCatalogueProfileRepository(ICatalogueProfileRepository):
    """SQLite-backed export catalogue profiles scoped to license customer."""

    def __init__(self, db_context: DatabaseContext) -> None:
        self._db = db_context

    @staticmethod
    def _row_to_master(row: sqlite3.Row):
        from src.services.catalogue_master_service import CatalogueMaster

        return CatalogueMaster(
            id=row["id"],
            display_name=row["display_name"],
            company_name=row["company_name"] or "",
            logo_path=row["logo_path"] or "",
            email=row["email"] or "",
            phone=row["phone"] or "",
            website=row["website"] or "",
            address=row["address"] or "",
            default_pdf_folder=row["default_pdf_folder"] or "",
            include_search_image=bool(row["include_search_image"]),
            include_image_path=bool(row["include_image_path"]),
            export_only_selected=bool(row["export_only_selected"]),
            watermark_text=row["watermark_text"] or "",
            max_results=int(row["max_results"] or 12),
        )

    @staticmethod
    def _master_values(profile) -> tuple:
        return (
            profile.display_name.strip(),
            profile.company_name.strip(),
            profile.logo_path.strip(),
            profile.email.strip(),
            profile.phone.strip(),
            profile.website.strip(),
            profile.address.strip(),
            profile.default_pdf_folder.strip(),
            int(profile.include_search_image),
            int(profile.include_image_path),
            int(profile.export_only_selected),
            profile.watermark_text.strip(),
            max(1, min(100, int(profile.max_results))),
        )

    def list_for_customer(self, license_customer_name: str):
        customer = license_customer_name.strip()
        if not customer:
            return []
        query = """
            SELECT * FROM catalogue_profiles
            WHERE license_customer_name = ?
            ORDER BY updated_at DESC, created_at DESC;
        """
        try:
            with self._db.session() as conn:
                rows = conn.execute(query, (customer,)).fetchall()
                return [self._row_to_master(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to list catalogue profiles: {e}")
            return []

    def get_by_id(self, license_customer_name: str, profile_id: str):
        customer = license_customer_name.strip()
        if not customer or not profile_id:
            return None
        query = """
            SELECT * FROM catalogue_profiles
            WHERE license_customer_name = ? AND id = ?;
        """
        try:
            with self._db.session() as conn:
                row = conn.execute(query, (customer, profile_id)).fetchone()
                return self._row_to_master(row) if row else None
        except sqlite3.Error as e:
            logger.error(f"Failed to get catalogue profile: {e}")
            return None

    def find_by_display_name(
        self,
        license_customer_name: str,
        display_name: str,
        *,
        exclude_id: Optional[str] = None,
    ):
        customer = license_customer_name.strip()
        target = " ".join(display_name.strip().split()).casefold()
        if not customer or not target:
            return None
        for master in self.list_for_customer(customer):
            if exclude_id and master.id == exclude_id:
                continue
            if " ".join(master.display_name.strip().split()).casefold() == target:
                return master
        return None

    def add(self, license_customer_name: str, profile):
        customer = license_customer_name.strip()
        if not customer:
            raise ValueError("Licensed customer name is required to save export profiles.")
        values = self._master_values(profile)
        query = """
            INSERT INTO catalogue_profiles (
                id, license_customer_name, display_name, company_name, logo_path,
                email, phone, website, address, default_pdf_folder,
                include_search_image, include_image_path, export_only_selected,
                watermark_text, max_results
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        try:
            with self._db.session() as conn:
                conn.execute(
                    query,
                    (profile.id, customer, *values),
                )
                conn.commit()
            return profile
        except sqlite3.IntegrityError as e:
            raise ValueError(
                f'A profile named "{profile.display_name.strip()}" already exists. '
                "Each customer can have only one profile."
            ) from e
        except sqlite3.Error as e:
            logger.error(f"Failed to add catalogue profile: {e}")
            raise RuntimeError(f"Database error saving export profile: {e}") from e

    def update(self, license_customer_name: str, profile):
        customer = license_customer_name.strip()
        if not customer:
            raise ValueError("Licensed customer name is required to save export profiles.")
        values = self._master_values(profile)
        query = """
            UPDATE catalogue_profiles
            SET display_name = ?, company_name = ?, logo_path = ?,
                email = ?, phone = ?, website = ?, address = ?,
                default_pdf_folder = ?, include_search_image = ?,
                include_image_path = ?, export_only_selected = ?,
                watermark_text = ?, max_results = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE license_customer_name = ? AND id = ?;
        """
        try:
            with self._db.session() as conn:
                cursor = conn.execute(
                    query,
                    (*values, customer, profile.id),
                )
                if cursor.rowcount == 0:
                    raise KeyError(f"Profile not found: {profile.id}")
                conn.commit()
            return profile
        except sqlite3.IntegrityError as e:
            raise ValueError(
                f'A profile named "{profile.display_name.strip()}" already exists. '
                "Each customer can have only one profile."
            ) from e
        except sqlite3.Error as e:
            logger.error(f"Failed to update catalogue profile: {e}")
            raise RuntimeError(f"Database error updating export profile: {e}") from e

    def delete(self, license_customer_name: str, profile_id: str) -> None:
        customer = license_customer_name.strip()
        if not customer:
            return
        try:
            with self._db.session() as conn:
                conn.execute(
                    """
                    DELETE FROM catalogue_profiles
                    WHERE license_customer_name = ? AND id = ?;
                    """,
                    (customer, profile_id),
                )
                prefs = conn.execute(
                    """
                    SELECT last_selected_id FROM catalogue_profile_prefs
                    WHERE license_customer_name = ?;
                    """,
                    (customer,),
                ).fetchone()
                if prefs and prefs["last_selected_id"] == profile_id:
                    conn.execute(
                        "DELETE FROM catalogue_profile_prefs WHERE license_customer_name = ?;",
                        (customer,),
                    )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to delete catalogue profile: {e}")
            raise RuntimeError(f"Database error deleting export profile: {e}") from e

    def get_last_selected_id(self, license_customer_name: str) -> Optional[str]:
        customer = license_customer_name.strip()
        if not customer:
            return None
        try:
            with self._db.session() as conn:
                row = conn.execute(
                    """
                    SELECT last_selected_id FROM catalogue_profile_prefs
                    WHERE license_customer_name = ?;
                    """,
                    (customer,),
                ).fetchone()
                if row and row["last_selected_id"]:
                    return str(row["last_selected_id"])
        except sqlite3.Error as e:
            logger.error(f"Failed to read catalogue profile prefs: {e}")
        return None

    def set_last_selected_id(
        self, license_customer_name: str, profile_id: Optional[str]
    ) -> None:
        customer = license_customer_name.strip()
        if not customer:
            return
        try:
            with self._db.session() as conn:
                if profile_id:
                    conn.execute(
                        """
                        INSERT INTO catalogue_profile_prefs (license_customer_name, last_selected_id)
                        VALUES (?, ?)
                        ON CONFLICT(license_customer_name) DO UPDATE SET
                            last_selected_id = excluded.last_selected_id;
                        """,
                        (customer, profile_id),
                    )
                else:
                    conn.execute(
                        "DELETE FROM catalogue_profile_prefs WHERE license_customer_name = ?;",
                        (customer,),
                    )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to save catalogue profile prefs: {e}")

    def count_for_customer(self, license_customer_name: str) -> int:
        customer = license_customer_name.strip()
        if not customer:
            return 0
        try:
            with self._db.session() as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS count FROM catalogue_profiles
                    WHERE license_customer_name = ?;
                    """,
                    (customer,),
                ).fetchone()
                return int(row["count"]) if row else 0
        except sqlite3.Error as e:
            logger.error(f"Failed to count catalogue profiles: {e}")
            return 0
