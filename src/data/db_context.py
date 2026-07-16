"""
Database context manager for TileVision AI.

Manages SQLite connection pooling/sessions, table initialization, and schema migrations.
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from src.data.db_protection import prepare_working_database

logger = logging.getLogger("tilevision.data.db_context")


class DatabaseContext:
    """
    Manages connections and schema initialization for the local SQLite database.
    
    Provides thread-safe access helper context managers for transactions.
    """

    def __init__(self, db_path: str) -> None:
        """
        Initialize the database context.

        Args:
            db_path: Absolute path to the SQLite database file.
        """
        self._db_path = Path(db_path)
        # Ensure parent directories exist
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        prepare_working_database(self._db_path)
        self.initialize_schema()

    @property
    def db_path(self) -> Path:
        """Absolute path to the SQLite database file (Task D: Settings, Database Size card)."""
        return self._db_path

    def initialize_schema(self) -> None:
        """Create database tables and indexes if they do not exist."""
        queries = [
            """
            CREATE TABLE IF NOT EXISTS tiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                file_name TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                dimensions TEXT NOT NULL,
                brand TEXT DEFAULT 'Unknown',
                category TEXT DEFAULT 'Unknown',
                color TEXT DEFAULT 'Unknown',
                size TEXT DEFAULT 'Unknown',
                product_code TEXT DEFAULT 'Unknown',
                width INTEGER DEFAULT 0,
                height INTEGER DEFAULT 0,
                sha256_hash TEXT DEFAULT '',
                perceptual_hash TEXT DEFAULT '',
                embedding_id INTEGER,
                embedding_blob BLOB,
                embedding_dimension INTEGER DEFAULT 1024,
                embedding_model TEXT DEFAULT 'facebook/dinov2-large',
                color_histogram BLOB,
                texture_histogram BLOB,
                edge_histogram BLOB,
                pattern_features BLOB,
                dominant_r INTEGER DEFAULT 0,
                dominant_g INTEGER DEFAULT 0,
                dominant_b INTEGER DEFAULT 0,
                feature_version INTEGER DEFAULT 0,
                pattern_feature_version INTEGER DEFAULT 0,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_indexed INTEGER DEFAULT 0
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS license_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                license_key TEXT NOT NULL,
                activated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at DATE,
                customer_name TEXT,
                hardware_hash TEXT NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS indexed_folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                folder_path TEXT NOT NULL UNIQUE,
                last_indexed_at TIMESTAMP,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_image_path TEXT NOT NULL,
                query_thumbnail_path TEXT,
                result_count INTEGER NOT NULL DEFAULT 0,
                elapsed_seconds REAL,
                searched_at TIMESTAMP NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now'))
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                activity_type TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now'))
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_tiles_file_name ON tiles(file_name);",
            "CREATE INDEX IF NOT EXISTS idx_search_history_searched_at ON search_history(searched_at);",
            "CREATE INDEX IF NOT EXISTS idx_activity_log_created_at ON activity_log(created_at);"
        ]

        logger.info(f"Initializing SQLite schema at: {self._db_path}")
        try:
            with self.session() as conn:
                cursor = conn.cursor()

                for query in queries:
                    cursor.execute(query)

                self._migrate_schema(conn)

                conn.commit()
            logger.info("Database schema initialized successfully.")
        except sqlite3.Error as e:
            logger.critical(f"Failed to initialize database schema: {e}")
            raise RuntimeError(f"Database initialization failure: {e}") from e

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """
        Migrate existing databases to the latest schema.

        Safe to execute on every application startup.
        """

        logger.info("Checking database schema...")

        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(tiles)")
        existing_columns = {
            row["name"] if isinstance(row, sqlite3.Row) else row[1]
            for row in cursor.fetchall()
        }

        migrations = {
            "embedding_blob": "BLOB",
            "embedding_dimension": "INTEGER DEFAULT 1024",
            "embedding_model": "TEXT DEFAULT 'facebook/dinov2-large'",
            "color_histogram": "BLOB",
            "texture_histogram": "BLOB",
            "edge_histogram": "BLOB",
            "pattern_features": "BLOB",
            "dominant_r": "INTEGER DEFAULT 0",
            "dominant_g": "INTEGER DEFAULT 0",
            "dominant_b": "INTEGER DEFAULT 0",
            "feature_version": "INTEGER DEFAULT 0",
            "pattern_feature_version": "INTEGER DEFAULT 0",
            "file_mtime": "REAL DEFAULT 0",
        }

        for column, definition in migrations.items():

            if column in existing_columns:
                continue

            logger.info("Adding column: %s", column)

            cursor.execute(
                f"""
                ALTER TABLE tiles
                ADD COLUMN {column} {definition}
                """
            )

        logger.info("Database schema is up to date.")
        
    @contextmanager
    def session(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for acquiring a SQLite connection.
        
        Handles committing transactions and closing connections cleanly.
        Yields a sqlite3.Connection object.
        """
        # check_same_thread=False is allowed if we guarantee single-thread access per session.
        # SQLite handles file locking automatically.
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=30.0,  # 30-second timeout for busy locks
            check_same_thread=False
        )
        # Enable foreign keys and row factory for dict-like rows if desired
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        
        try:
            yield conn
        except Exception as e:
            logger.error(f"Database session error, rolling back: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
