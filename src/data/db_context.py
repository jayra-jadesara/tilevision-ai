"""
Database context manager for TileVision AI.

Manages SQLite connection pooling/sessions, table initialization, and schema migrations.
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

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
        self.initialize_schema()

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
            "CREATE INDEX IF NOT EXISTS idx_tiles_file_name ON tiles(file_name);"
        ]

        logger.info(f"Initializing SQLite schema at: {self._db_path}")
        try:
            with self.session() as conn:
                cursor = conn.cursor()
                for query in queries:
                    cursor.execute(query)
                conn.commit()
            logger.info("Database schema initialized successfully.")
        except sqlite3.Error as e:
            logger.critical(f"Failed to initialize database schema: {e}")
            raise RuntimeError(f"Database initialization failure: {e}") from e

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
