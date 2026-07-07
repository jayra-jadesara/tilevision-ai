"""
Image indexing use case module for TileVision AI.

Coordinates scanning file directories, validating image structures, generating cached
thumbnails, generating embeddings via the AI model, and indexing into the FAISS database.
Supports incremental indexing, perceptual hashing, SHA-256 matching, and cooperative thread execution.
"""

import logging
from pathlib import Path
import threading
import time
from typing import Callable, List, Optional, Tuple

from src.core.models import TileImage
from src.data.repository_interface import IImageRepository
from src.ai.embedder import OpenCLIPEmbedder
from src.ai.vector_index import FaissIndexManager
from src.utils.image_utils import (
    validate_image,
    get_image_metadata,
    generate_thumbnail,
    compute_sha256,
    compute_dhash,
)

logger = logging.getLogger("tilevision.core.use_cases.index_images")


def parse_filename_metadata(stem: str) -> Tuple[str, str, str, str, str]:
    """
    Parse brand, category, color, size, and product code from filename stem
    assuming the format: Brand_Category_Color_Size_ProductCode.
    
    If the name does not match this structure, falls back gracefully.
    """
    parts = stem.split("_")
    brand = parts[0] if len(parts) > 0 and parts[0] else "Unknown"
    category = parts[1] if len(parts) > 1 and parts[1] else "Unknown"
    color = parts[2] if len(parts) > 2 and parts[2] else "Unknown"
    size = parts[3] if len(parts) > 3 and parts[3] else "Unknown"
    product_code = parts[4] if len(parts) > 4 and parts[4] else "Unknown"

    return (
        brand.strip(),
        category.strip(),
        color.strip(),
        size.strip(),
        product_code.strip(),
    )


class IndexImagesUseCase:
    """
    Use case to index single image files or whole directories recursively.
    Supports pausing, resuming, and canceling operations cooperatively.
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
        self._supported_extensions = {".png", ".jpg", ".jpeg", ".webp"}

        # Ensure thumbnail directory exists
        self._thumbnail_dir.mkdir(parents=True, exist_ok=True)

    def index_single_file(self, file_path: Path) -> int:
        """
        Index a single tile image.
        
        Validates, computes SHA256 and perceptual hashes, extracts metadata,
        generates thumbnail, extracts features, and updates FAISS and SQLite.

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

        # Compute hashes and dimensions
        file_size, dimensions = get_image_metadata(resolved_path)
        sha256_hash = compute_sha256(resolved_path)
        perceptual_hash = compute_dhash(resolved_path)
        file_name = resolved_path.name

        # Parse width/height from dimensions string "WxH"
        width, height = 0, 0
        if "x" in dimensions:
            try:
                w_str, h_str = dimensions.split("x")
                width, height = int(w_str), int(h_str)
            except ValueError:
                pass

        # Parse naming metadata
        brand, category, color, size, product_code = parse_filename_metadata(resolved_path.stem)

        # 1. Store/update metadata record in SQLite
        # Note: We temporarily insert without embedding_id, get the ID, and then update it.
        tile = TileImage(
            file_path=str(resolved_path),
            file_name=file_name,
            file_size=file_size,
            dimensions=dimensions,
            brand=brand,
            category=category,
            color=color,
            size=size,
            product_code=product_code,
            width=width,
            height=height,
            sha256_hash=sha256_hash,
            perceptual_hash=perceptual_hash,
            is_indexed=False,
        )
        
        # Save record to get the database auto-increment ID
        db_id = self._repo.add(tile)
        
        # Keep embedding ID identical to the database row ID
        tile.id = db_id
        tile.embedding_id = db_id

        try:
            # 2. Generate and cache thumbnail
            generate_thumbnail(resolved_path, self._thumbnail_dir)

            # 3. Extract embedding vector
            logger.info(f"Extracting embedding features for tile ID {db_id}: {file_name}")
            embedding = self._embedder.get_embedding(str(resolved_path))

            # 4. Insert embedding into FAISS index
            logger.info(f"Indexing vector into FAISS for ID {db_id}")
            self._index.add_vectors([db_id], [embedding])

            # 5. Update SQLite record with embedding_id and mark as successfully indexed
            tile.is_indexed = True
            self._repo.add(tile)
            
            return db_id
        except Exception as e:
            logger.error(f"Failed to complete indexing pipeline for file {resolved_path}: {e}")
            # Clean up the record or mark it as failed if needed, but we raise to let worker handle
            raise RuntimeError(f"Indexing pipeline failed for file: {file_name}") from e

    def scan_and_index_directory(
        self,
        directory_path: Path,
        progress_callback: Optional[Callable[[int, int, str, float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        pause_event: Optional[threading.Event] = None,
    ) -> Tuple[int, int, bool]:
        """
        Scan a directory recursively, comparing file states to skip already indexed images,
        and batch index all new or modified image files.
        Supports cooperative thread pause/resume/cancel.

        Args:
            directory_path: Root folder path to scan.
            progress_callback: Optional callback receiving (processed_count, total_count, current_filename, eta_seconds).
            cancel_event: Threading event to stop the scan loop.
            pause_event: Threading event to temporarily pause the scan loop.

        Returns:
            A tuple of (successfully_indexed_count, skipped_count, is_completed_bool).
        """
        root = Path(directory_path).resolve()
        if not root.exists() or not root.is_dir():
            logger.error(f"Invalid directory path provided for scan: {root}")
            return 0, 0, False

        logger.info(f"Scanning directory for new tiles: {root}")

        # Find all files with supported extensions
        all_files: List[Path] = []
        for ext in self._supported_extensions:
            all_files.extend(root.rglob(f"*{ext}"))
            all_files.extend(root.rglob(f"*{ext.upper()}"))
            
        all_files = sorted(list(set(all_files)))
        total_files = len(all_files)
        
        logger.info(f"Found {total_files} potential image files in {root}")

        indexed_count = 0
        skipped_count = 0
        is_completed = True

        self._index.load_index()

        start_time = time.time()
        processed_count = 0

        for file_path in all_files:
            # 1. Cooperative Pause check
            if pause_event and pause_event.is_set():
                logger.info("Indexing worker paused. Waiting for resume...")
                while pause_event.is_set():
                    # Poll cancel during pause
                    if cancel_event and cancel_event.is_set():
                        break
                    time.sleep(0.2)

            # 2. Cooperative Cancel check
            if cancel_event and cancel_event.is_set():
                logger.warning("Indexing worker canceled by user request. Stopping...")
                is_completed = False
                break

            resolved_path = file_path.resolve()
            file_name = resolved_path.name

            # Calculate ETA
            elapsed = time.time() - start_time
            if processed_count > 0:
                avg_time = elapsed / processed_count
                remaining_files = total_files - processed_count
                eta = avg_time * remaining_files
            else:
                eta = 0.0

            # Emit progress update before processing
            if progress_callback:
                progress_callback(processed_count, total_files, file_name, eta)

            processed_count += 1

            try:
                # 3. Incremental Index Check
                existing_record = self._repo.get_by_path(str(resolved_path))
                
                if existing_record and existing_record.is_indexed:
                    # Verify SHA-256 hash to determine if file is identical
                    current_sha = compute_sha256(resolved_path)
                    if current_sha == existing_record.sha256_hash:
                        logger.debug(f"Skipping unchanged tile: {file_name}")
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
            progress_callback(total_files, total_files, "Done", 0.0)

        return indexed_count, skipped_count, is_completed
