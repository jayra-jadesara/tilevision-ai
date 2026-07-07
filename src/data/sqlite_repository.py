"""
SQLite implementation of the repository interfaces for TileVision AI.

Handles database transactions and data conversions between SQLite Rows
and domain models.
"""

from datetime import datetime
import logging
import sqlite3
from typing import List, Optional

from src.core.models import TileImage, LicenseInfo
from src.data.db_context import DatabaseContext
from src.data.repository_interface import IImageRepository, ILicenseRepository

logger = logging.getLogger("tilevision.data.sqlite_repository")


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
        created_time = None
        if row["created_time"]:
            try:
                created_time = datetime.fromisoformat(row["created_time"])
            except ValueError:
                try:
                    created_time = datetime.strptime(row["created_time"], "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    logger.warning(f"Could not parse created_time timestamp: {row['created_time']}")

        updated_time = None
        if row["updated_time"]:
            try:
                updated_time = datetime.fromisoformat(row["updated_time"])
            except ValueError:
                try:
                    updated_time = datetime.strptime(row["updated_time"], "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    logger.warning(f"Could not parse updated_time timestamp: {row['updated_time']}")

        return TileImage(
            id=row["id"],
            file_path=row["file_path"],
            file_name=row["file_name"],
            file_size=row["file_size"],
            dimensions=row["dimensions"],
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
            file_path, file_name, file_size, dimensions,
            brand, category, color, size, product_code,
            width, height, sha256_hash, perceptual_hash, embedding_id, is_indexed, updated_time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(file_path) DO UPDATE SET
            file_name=excluded.file_name,
            file_size=excluded.file_size,
            dimensions=excluded.dimensions,
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
            is_indexed=excluded.is_indexed,
            updated_time=CURRENT_TIMESTAMP
        RETURNING id;
        """
        try:
            with self._db.session() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    query,
                    (
                        tile.file_path,
                        tile.file_name,
                        tile.file_size,
                        tile.dimensions,
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
        activated_date = None
        if row["activated_date"]:
            try:
                activated_date = datetime.fromisoformat(row["activated_date"])
            except ValueError:
                try:
                    activated_date = datetime.strptime(row["activated_date"], "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    logger.warning(f"Could not parse activated_date timestamp: {row['activated_date']}")

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
