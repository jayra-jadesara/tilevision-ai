"""
Image indexing use case module for TileVision AI.

Coordinates scanning file directories, validating image structures, generating cached
thumbnails, generating embeddings via the AI model, and indexing into the FAISS database.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.core.models import TileImage
from src.data.repository_interface import IImageRepository
from src.ai.embedder import OpenCLIPEmbedder
from src.ai.vector_index import FaissIndexManager
from src.utils.image_utils import validate_image, get_image_metadata, generate_thumbnail

logger = logging.getLogger("tilevision.core.use_cases.index_images")


class IndexImagesUseCase:
    """
    Use case to index single image files or whole directories recursively.
    """

    def __init__(
        self,
        image_repository: IImageRepository,
        embedder: OpenCLIPEmbedder,
        vector_index: FaissIndexManager,
        thumbnail_dir: str,
    ) -> None:
        """
        Initialize the indexing use case.

        Args:
            image_repository: Repository interface for SQLite.
            embedder: CLIP model embedder wrapper.
            vector_index: FAISS index manager wrapper.
            thumbnail_dir: Folder path where thumbnails are cached.
        """
        self._repo = image_repository
        self._embedder = embedder
        self._index = vector_index
        self._thumbnail_dir = Path(thumbnail_dir)
        self._supported_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

        # Ensure thumbnail directory exists
        self._thumbnail_dir.mkdir(parents=True, exist_ok=True)

    def index_single_file(self, file_path: Path) -> int:
        """
        Index a single tile image.
        
        Validates, records metadata in SQL, generates a thumbnail, extracts OpenCLIP
        features, inserts the vector into FAISS, and marks the image as indexed.

        Args:
            file_path: Absolute path to the tile image file.

        Returns:
            The database primary key ID of the indexed tile.

        Raises:
            ValueError: If the file is not a valid image.
            RuntimeError: If embedding extraction or FAISS storage fails.
        """
        resolved_path = file_path.resolve()
        if not validate_image(resolved_path):
            raise ValueError(f"File is not a valid or accessible image: {resolved_path}")

        file_size, dimensions = get_image_metadata(resolved_path)
        file_name = resolved_path.name

        # 1. Store or update metadata in database
        tile = TileImage(
            file_path=str(resolved_path),
            file_name=file_name,
            file_size=file_size,
            dimensions=dimensions,
            is_indexed=False,
        )
        
        # Save record to get the database auto-increment ID
        db_id = self._repo.add(tile)
        
        try:
            # 2. Generate and cache thumbnail
            generate_thumbnail(resolved_path, self._thumbnail_dir)

            # 3. Extract embedding vector
            logger.info(f"Extracting embedding features for tile ID {db_id}: {file_name}")
            embedding = self._embedder.get_embedding(str(resolved_path))

            # 4. Insert embedding into FAISS index
            logger.info(f"Indexing vector into FAISS for ID {db_id}")
            self._index.add_vectors([db_id], [embedding])

            # 5. Mark as successfully indexed in SQLite database
            self._repo.mark_as_indexed(db_id, True)
            
            return db_id
        except Exception as e:
            logger.error(f"Failed to complete indexing pipeline for file {resolved_path}: {e}")
            raise RuntimeError(f"Indexing pipeline failed for file: {file_name}") from e

    def scan_and_index_directory(
        self,
        directory_path: Path,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[int, int]:
        """
        Scan a directory recursively, comparing file states to skip already indexed images,
        and batch index all new or modified image files.

        Args:
            directory_path: Root folder path to scan.
            progress_callback: Optional callback receiving (processed_count, total_count, current_filename).

        Returns:
            A tuple of (successfully_indexed_count, skipped_count).
        """
        root = Path(directory_path).resolve()
        if not root.exists() or not root.is_dir():
            logger.error(f"Invalid directory path provided for scan: {root}")
            return 0, 0

        logger.info(f"Scanning directory for new tiles: {root}")

        # Find all files with supported extensions
        all_files: List[Path] = []
        for ext in self._supported_extensions:
            all_files.extend(root.rglob(f"*{ext}"))
            all_files.extend(root.rglob(f"*{ext.upper()}"))
            
        # Deduplicate files (rglob on case-insensitive can return duplicates on some FS)
        all_files = sorted(list(set(all_files)))
        total_files = len(all_files)
        
        logger.info(f"Found {total_files} potential image files in {root}")

        indexed_count = 0
        skipped_count = 0

        # Load models/index if not already done
        self._index.load_index()

        for idx, file_path in enumerate(all_files):
            resolved_path = file_path.resolve()
            file_name = resolved_path.name

            # Emit progress update before processing
            if progress_callback:
                progress_callback(idx, total_files, file_name)

            try:
                # Cache checking logic
                existing_record = self._repo.get_by_path(str(resolved_path))
                
                if existing_record and existing_record.is_indexed:
                    # Check if file has been modified (size comparison)
                    try:
                        current_size = resolved_path.stat().st_size
                    except OSError:
                        current_size = -1
                        
                    if current_size == existing_record.file_size:
                        logger.debug(f"Skipping unmodified tile: {file_name}")
                        skipped_count += 1
                        continue

                # Run indexing pipeline
                self.index_single_file(resolved_path)
                indexed_count += 1
            except Exception as e:
                logger.error(f"Error indexing file during folder scan: {resolved_path}. Error: {e}")
                # Continue scanning other files in case of a single corrupt image

        # Commit final index state back to disk
        if indexed_count > 0:
            self._index.save_index()
            logger.info(f"Index directory complete. Indexed: {indexed_count}, Skipped: {skipped_count}")

        # Emit final progress update
        if progress_callback:
            progress_callback(total_files, total_files, "Done")

        return indexed_count, skipped_count
