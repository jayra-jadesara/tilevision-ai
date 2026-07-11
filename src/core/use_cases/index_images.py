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

from src.core.models import TileImage, ScanResult, IndexedFolderState
from src.data.repository_interface import IImageRepository, IIndexedFolderRepository
from src.ai.embedder import OpenCLIPEmbedder
from src.ai.vector_index import FaissIndexManager
from src.utils.image_utils import (
    validate_image,
    get_image_metadata,
    generate_thumbnail,
    compute_sha256,
    compute_dhash,
    SUPPORTED_IMAGE_EXTENSIONS,
)

logger = logging.getLogger("tilevision.core.use_cases.index_images")

# Persist the FAISS index to disk every N processed files during a folder
# scan (rather than after every single file, which is a heavy full-index
# disk write and would dominate indexing time on large catalogs).
_CHECKPOINT_INTERVAL = 25


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
        folder_repository: Optional[IIndexedFolderRepository] = None,
    ) -> None:
        """
        Initialize the indexing use case.

        Args:
            image_repository: Repository interface for SQLite.
            embedder: CLIP model embedder wrapper.
            vector_index: FAISS index manager wrapper.
            thumbnail_dir: Folder path where thumbnails are cached.
            folder_repository: Optional repository for recording which
                folders have been indexed (Task 1: Persistent Indexed
                Folder). If omitted, scan_and_index_directory() still
                works but won't persist folder state across restarts —
                kept optional so existing callers/tests that construct
                this use case without it keep working unchanged.
        """
        self._repo = image_repository
        self._embedder = embedder
        self._index = vector_index
        self._thumbnail_dir = Path(thumbnail_dir)
        self._supported_extensions = set(SUPPORTED_IMAGE_EXTENSIONS)
        self._folder_repo = folder_repository

        # Ensure thumbnail directory exists
        self._thumbnail_dir.mkdir(parents=True, exist_ok=True)

    def index_single_file(self, file_path: Path, persist: bool = True) -> int:
        """
        Index a single tile image.
        
        Validates, computes SHA256 and perceptual hashes, extracts metadata,
        generates thumbnail, extracts features, and updates FAISS and SQLite.

        Args:
            file_path: Absolute path to the tile image file.
            persist: If True (default), writes the FAISS index to disk
                immediately after indexing this file. Folder scans process
                many files in a loop and pass False, checkpointing the index
                to disk periodically instead (see scan_and_index_directory).

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

            # 4. Insert/replace embedding into FAISS index. update_vectors()
            #    removes any existing vector for this id first, so a changed
            #    file's stale embedding never lingers alongside the fresh one.
            logger.info(f"Indexing vector into FAISS for ID {db_id}")
            self._index.update_vectors([db_id], [embedding], persist=persist)

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
    ) -> ScanResult:
        """
        Scan a directory recursively and perform a "smart" incremental
        index (Task 2): only new or modified files are (re-)embedded;
        unchanged files are skipped entirely; files that were previously
        indexed but no longer exist on disk are detected and removed from
        both FAISS and SQLite. Supports cooperative thread pause/resume/cancel.

        On successful completion, records this folder as indexed (Task 1:
        Persistent Indexed Folder) so the Index page can restore its state
        on the next app launch without requiring a re-scan.

        Args:
            directory_path: Root folder path to scan.
            progress_callback: Optional callback receiving (processed_count, total_count, current_filename, eta_seconds).
            cancel_event: Threading event to stop the scan loop.
            pause_event: Threading event to temporarily pause the scan loop.

        Returns:
            A ScanResult with the new/modified/deleted/skipped breakdown.
        """
        root = Path(directory_path).resolve()
        if not root.exists() or not root.is_dir():
            logger.error(f"Invalid directory path provided for scan: {root}")
            return ScanResult(is_completed=False)

        logger.info(f"Scanning directory for new tiles: {root}")

        # Find all files with supported extensions
        all_files: List[Path] = []
        for ext in self._supported_extensions:
            all_files.extend(root.rglob(f"*{ext}"))
            all_files.extend(root.rglob(f"*{ext.upper()}"))

        all_files = sorted(list(set(all_files)))
        total_files = len(all_files)

        logger.info(f"Found {total_files} potential image files in {root}")

        new_count = 0
        modified_count = 0
        skipped_count = 0
        failed_count = 0
        is_completed = True

        self._index.load_index()

        start_time = time.time()
        processed_count = 0

        # Track every path we actually see on disk this scan, so we can
        # detect deletions afterward (any previously-indexed tile under
        # this folder whose path isn't in this set was removed from disk).
        paths_seen_on_disk: set = set()

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
            paths_seen_on_disk.add(str(resolved_path))

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
                # 3. Incremental Index Check: distinguish unchanged (skip),
                #    modified (re-embed, was already a known file), and new
                #    (re-embed, never seen before).
                existing_record = self._repo.get_by_path(str(resolved_path))
                was_previously_indexed = bool(existing_record and existing_record.is_indexed)

                if was_previously_indexed:
                    current_sha = compute_sha256(resolved_path)
                    if current_sha == existing_record.sha256_hash:
                        logger.debug(f"Skipping unchanged tile: {file_name}")
                        skipped_count += 1
                        continue

                # Run indexing pipeline. persist=False: avoid a full FAISS
                # index disk write after every single file — see checkpoint
                # flush below instead.
                self.index_single_file(resolved_path, persist=False)

                if was_previously_indexed:
                    modified_count += 1
                else:
                    new_count += 1
            except Exception as e:
                failed_count += 1
                logger.error(f"Error indexing file during folder scan: {resolved_path}. Error: {e}")
                # Continue scanning other files in case of a single corrupt image

            # Checkpoint: flush FAISS to disk periodically so a crash, power
            # loss, or long-running scan never loses more than one batch of
            # progress (also matters if the user leaves it running overnight).
            if (new_count + modified_count) > 0 and processed_count % _CHECKPOINT_INTERVAL == 0:
                self._index.save_index()

        # 4. Deletion detection (Task 2): any tile previously indexed under
        #    this folder that we did NOT see on disk during this scan was
        #    deleted/moved away — remove it from FAISS and SQLite so stale
        #    entries don't linger in search results forever. Skipped
        #    entirely if the scan was cancelled partway through, since a
        #    partial file listing would incorrectly look like mass deletion.
        deleted_count = 0
        if is_completed:
            deleted_count = self._remove_deleted_tiles(root, paths_seen_on_disk)

        # Always flush any pending vector additions/removals before
        # returning, whether the scan completed, was cancelled, or hit
        # errors along the way — partial progress must never be lost.
        if (new_count + modified_count) > 0 or deleted_count > 0:
            self._index.save_index()
            logger.info(
                f"Index directory complete. New: {new_count}, Modified: {modified_count}, "
                f"Deleted: {deleted_count}, Skipped: {skipped_count}, Failed: {failed_count}"
            )

        # Record this folder as indexed (Task 1) so the Index page can
        # restore its state on next startup without re-scanning. Only
        # recorded on a completed (non-cancelled) scan.
        if is_completed and self._folder_repo is not None:
            self._folder_repo.record_folder_indexed(str(root))

        elapsed_total = time.time() - start_time

        # "Time saved" (Task 2): estimate how much longer this scan would
        # have taken if every skipped (unchanged) file had been re-embedded
        # too, based on this scan's own observed average processing time.
        # Falls back to a reasonable fixed estimate when nothing was
        # actually processed this scan (e.g. everything was already
        # up to date), since there's no real per-file timing to derive from.
        _FALLBACK_SECONDS_PER_IMAGE = 0.35
        actually_processed = new_count + modified_count
        avg_seconds_per_image = (
            elapsed_total / actually_processed if actually_processed > 0 else _FALLBACK_SECONDS_PER_IMAGE
        )
        time_saved = skipped_count * avg_seconds_per_image

        # Emit final progress update
        if progress_callback:
            progress_callback(total_files, total_files, "Done", 0.0)

        return ScanResult(
            new_count=new_count,
            modified_count=modified_count,
            deleted_count=deleted_count,
            skipped_count=skipped_count,
            failed_count=failed_count,
            total_files_scanned=total_files,
            is_completed=is_completed,
            elapsed_seconds=elapsed_total,
            time_saved_seconds=time_saved,
        )

    def get_last_indexed_folder_status(self) -> Optional[IndexedFolderState]:
        """
        Retrieve the most recently indexed folder along with a live count
        of currently-indexed tiles under it, for restoring the Index page's
        state on application startup (Task 1: Persistent Indexed Folder).

        Returns:
            An IndexedFolderState with indexed_image_count populated, or
            None if no folder has ever been indexed (or no folder
            repository was configured for this use case instance).
        """
        if self._folder_repo is None:
            return None

        state = self._folder_repo.get_last_indexed_folder()
        if state is None:
            return None

        state.indexed_image_count = self._count_indexed_tiles_under(state.folder_path)
        return state

    def _count_indexed_tiles_under(self, folder_path: str) -> int:
        """Live count of indexed tiles whose file_path falls under folder_path."""
        return sum(
            1 for tile in self._repo.get_all()
            if tile.file_path.startswith(folder_path) and tile.is_indexed
        )

    def _remove_deleted_tiles(self, folder_root: Path, paths_seen_on_disk: set) -> int:
        """
        Remove tiles from FAISS + SQLite that were previously indexed under
        `folder_root` but are no longer present on disk.

        Args:
            folder_root: The resolved folder path that was just scanned.
            paths_seen_on_disk: Every file path actually found during the
                scan (resolved, as strings).

        Returns:
            The number of tiles removed.
        """
        root_str = str(folder_root)
        deleted_count = 0

        for tile in self._repo.get_all():
            if not tile.file_path.startswith(root_str):
                continue  # not under the folder we just scanned
            if tile.file_path in paths_seen_on_disk:
                continue  # still present on disk

            if tile.id is None:
                continue

            try:
                if tile.embedding_id is not None:
                    self._index.remove_vectors([tile.embedding_id])
                self._repo.remove(tile.id)
                deleted_count += 1
                logger.info(f"Removed deleted file from index: {tile.file_path}")
            except Exception as e:
                logger.error(f"Failed to remove deleted tile {tile.file_path}: {e}")

        return deleted_count
