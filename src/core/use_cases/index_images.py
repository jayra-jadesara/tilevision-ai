"""
Image indexing use case module for TileVision AI.

Coordinates scanning file directories, validating image structures, generating cached
thumbnails, generating embeddings via the AI model, and indexing into the FAISS database.
Supports incremental indexing, perceptual hashing, SHA-256 matching, and cooperative thread execution.
"""

import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.config.indexing_performance import IndexingPerformanceConfig
from src.core.models import TileImage, ScanResult, IndexedFolderState
from src.data.repository_interface import IImageRepository, IIndexedFolderRepository
from src.ai.feature_extractor import FeatureExtractor
from src.ai.vector_index import FaissIndexManager
from src.utils.pipeline_timing import PipelineTimer
from src.utils.image_utils import (
    validate_image,
    get_image_metadata,
    generate_thumbnail,
    compute_sha256,
    compute_dhash,
    SUPPORTED_IMAGE_EXTENSIONS,
)

logger = logging.getLogger("tilevision.core.use_cases.index_images")

_SIZE_PATTERN = re.compile(r"(\d{2,4})\s*[xX×]\s*(\d{2,4})")
_COLOR_TOKENS = frozenset({
    "white", "black", "grey", "gray", "cream", "beige", "brown", "blue", "green",
    "red", "ivory", "bone", "taupe", "sand", "charcoal", "anthracite", "gold",
    "silver", "bronze", "copper", "walnut", "oak", "teak", "ebony", "onyx",
})
_CATEGORY_TOKENS = frozenset({
    "floor", "wall", "tile", "tiles", "ceramic", "porcelain", "marble", "granite",
    "wood", "stone", "terrazzo", "mosaic", "slate", "travertine", "cement",
    "concrete", "vinyl", "decor", "decorative",
})
# Placement tokens beat material tokens when both appear in a filename.
_CATEGORY_PRIORITY = (
    "floor",
    "wall",
    "tile",
    "tiles",
    "decor",
    "decorative",
    "ceramic",
    "porcelain",
    "marble",
    "granite",
    "wood",
    "stone",
    "terrazzo",
    "mosaic",
    "slate",
    "travertine",
    "cement",
    "concrete",
    "vinyl",
)


def _extract_size_from_stem(stem: str) -> str:
    match = _SIZE_PATTERN.search(stem)
    if not match:
        return "Unknown"
    return f"{match.group(1)}x{match.group(2)}"


def parse_filename_metadata(stem: str) -> Tuple[str, str, str, str, str]:
    """
    Parse brand, category, color, size, and product code from filename stem.

    Supports:
    - ``Brand_Category_Color_Size_ProductCode``
    - Descriptive hyphenated names such as ``cream-color-floor-tiles``
    """
    size = _extract_size_from_stem(stem)
    parts = [part.strip() for part in stem.split("_") if part.strip()]

    if len(parts) >= 2:
        brand = parts[0] or "Unknown"
        category = parts[1] if len(parts) > 1 else "Unknown"
        color = parts[2] if len(parts) > 2 else "Unknown"
        parsed_size = parts[3] if len(parts) > 3 else size
        product_code = parts[4] if len(parts) > 4 else "Unknown"
        if parsed_size and parsed_size.lower() != "unknown":
            size = parsed_size
        return (
            brand.strip(),
            category.strip(),
            color.strip(),
            size.strip(),
            product_code.strip(),
        )

    tokens = [token for token in re.split(r"[-\s]+", stem.lower()) if token]
    if tokens:
        found_color = next(
            (token.title() for token in tokens if token in _COLOR_TOKENS),
            "Unknown",
        )
        token_set = set(tokens)
        found_category = next(
            (
                token.title()
                for token in _CATEGORY_PRIORITY
                if token in token_set
            ),
            "Unknown",
        )
        return (
            "Unknown",
            found_category,
            found_color,
            size,
            stem[:80],
        )

    return ("Unknown", "Unknown", "Unknown", size, "Unknown")


@dataclass(slots=True)
class _PendingIndexItem:
    path: Path
    was_previously_indexed: bool


def _is_unchanged_indexed_file(
    path: Path,
    existing_record: Optional[TileImage],
    *,
    force: bool,
) -> bool:
    """
    Return True when an indexed file can be skipped without re-embedding.

    Uses size+mtime fingerprint first (no disk read). Falls back to SHA256
    for legacy rows that predate file_mtime storage.
    """
    if force or existing_record is None or not existing_record.is_indexed:
        return False

    stat = path.stat()
    if stat.st_size != existing_record.file_size:
        return False

    stored_mtime = float(existing_record.file_mtime or 0.0)
    if stored_mtime > 0 and abs(stat.st_mtime - stored_mtime) < 1e-6:
        return True

    if stored_mtime <= 0:
        current_sha = compute_sha256(path)
        return current_sha == existing_record.sha256_hash

    return False


def _max_file_bytes(items: List[_PendingIndexItem]) -> int:
    return max(item.path.stat().st_size for item in items)


class IndexImagesUseCase:
    """
    Use case to index single image files or whole directories recursively.
    Supports pausing, resuming, and canceling operations cooperatively.
    """

    def __init__(
        self,
        image_repository: IImageRepository,
        feature_extractor: FeatureExtractor,
        vector_index: FaissIndexManager,
        thumbnail_dir: str,
        folder_repository: Optional[IIndexedFolderRepository] = None,
        performance: Optional[IndexingPerformanceConfig] = None,
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
        self._feature_extractor = feature_extractor
        self._index = vector_index
        self._thumbnail_dir = Path(thumbnail_dir)
        self._supported_extensions = set(SUPPORTED_IMAGE_EXTENSIONS)
        self._folder_repo = folder_repository
        self._perf = performance or IndexingPerformanceConfig()

        # Ensure thumbnail directory exists
        self._thumbnail_dir.mkdir(parents=True, exist_ok=True)

    def _pending_batch_limit(self, items: List[_PendingIndexItem]) -> int:
        if not items:
            return self._perf.batch_size
        return self._perf.adaptive_batch_size(_max_file_bytes(items))

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
        timer = PipelineTimer("INDEX TIMING")

        with timer.measure("image_loading"):
            if not validate_image(resolved_path):
                raise ValueError(f"File is not a valid or accessible image: {resolved_path}")

            file_size, dimensions = get_image_metadata(resolved_path)
            sha256_hash = compute_sha256(resolved_path)
            perceptual_hash = compute_dhash(resolved_path)
            file_mtime = resolved_path.stat().st_mtime

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

        logger.info("Extracting AI features...")
        features = self._feature_extractor.extract(str(resolved_path))
        extract_timings = self._feature_extractor.last_timings
        timer.timings.record("preprocessing", extract_timings.preprocessing)
        timer.timings.record("dinov2", extract_timings.dinov2)
        timer.timings.record("descriptors", extract_timings.descriptors)

        # 1. Store/update metadata record in SQLite
        # Note: We temporarily insert without embedding_id, get the ID, and then update it.
        tile = TileImage(
            file_path=str(resolved_path),
            file_name=file_name,
            file_size=file_size,
            dimensions=dimensions,
            file_mtime=file_mtime,
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
            features=features
        )
        
        # Save record to get the database auto-increment ID
        with timer.measure("database"):
            db_id = self._repo.add(tile)
        
        # Keep embedding ID identical to the database row ID
        tile.id = db_id
        tile.embedding_id = db_id

        try:
            # 2. Generate and cache thumbnail
            generate_thumbnail(
                resolved_path,
                self._thumbnail_dir,
            )

            # 3. Reuse embedding already extracted above.
            # Do NOT run FeatureExtractor again.
            logger.info(
                f"Indexing vector into FAISS for ID {db_id}: {file_name}"
            )

            with timer.measure("faiss"):
                self._index.update_vectors(
                    [db_id],
                    [features.embedding.tolist()],
                    persist=persist,
                )

            # 4. Mark tile as successfully indexed
            tile.is_indexed = True

            with timer.measure("database"):
                self._repo.add(tile)

            timer.log_summary(log=logger)
            return db_id

        except Exception as e:
            logger.error(
                f"Failed to complete indexing pipeline for file "
                f"{resolved_path}: {e}"
            )
            raise RuntimeError(
                f"Indexing pipeline failed for file: {file_name}"
            ) from e

    def index_changed_file(self, file_path: Path, persist: bool = True) -> Optional[int]:
        """
        Index a file when it is new or changed; skip unchanged indexed files.

        Used by auto folder monitoring so in-place edits do not always
        re-embed unless size/mtime/hash changed.

        Returns:
            Database id when indexed, or None when skipped as unchanged.
        """
        resolved_path = file_path.resolve()
        existing_record = self._repo.get_by_path(str(resolved_path))
        if _is_unchanged_indexed_file(resolved_path, existing_record, force=False):
            logger.debug("Auto-monitor skipping unchanged file: %s", resolved_path.name)
            return None

        db_id = self.index_single_file(resolved_path, persist=persist)
        self._record_parent_folder(resolved_path)
        return db_id

    def remove_indexed_file(self, file_path: Path) -> bool:
        """Remove one indexed tile after its image file was deleted."""
        resolved = file_path.resolve()
        tile = self._repo.get_by_path(str(resolved))
        if tile is None or tile.id is None:
            return False

        try:
            if tile.embedding_id is not None:
                self._index.remove_vectors([tile.embedding_id])
                self._index.save_index()
            self._repo.remove(tile.id)
            logger.info("Removed deleted file from index: %s", resolved)
            return True
        except Exception as exc:
            logger.error("Failed to remove deleted tile %s: %s", resolved, exc)
            return False

    def _record_parent_folder(self, file_path: Path) -> None:
        if self._folder_repo is None:
            return
        self._folder_repo.record_folder_indexed(str(file_path.parent.resolve()))

    def _index_file_batch(
        self,
        items: List[_PendingIndexItem],
        persist: bool = True,
    ) -> None:
        """
        Index multiple files using batched DINOv2 inference and batch FAISS insert.
        """
        if not items:
            return

        timer = PipelineTimer("INDEX BATCH TIMING")
        path_strings = [str(item.path) for item in items]
        tiles: List[TileImage] = []

        with timer.measure("image_loading"):
            for item in items:
                resolved_path = item.path
                if not validate_image(resolved_path):
                    raise ValueError(
                        f"File is not a valid or accessible image: {resolved_path}"
                    )

                file_size, dimensions = get_image_metadata(resolved_path)
                sha256_hash = compute_sha256(resolved_path)
                perceptual_hash = compute_dhash(resolved_path)
                file_mtime = resolved_path.stat().st_mtime

                width, height = 0, 0
                if "x" in dimensions:
                    try:
                        w_str, h_str = dimensions.split("x")
                        width, height = int(w_str), int(h_str)
                    except ValueError:
                        pass

                brand, category, color, size, product_code = parse_filename_metadata(
                    resolved_path.stem
                )

                tiles.append(
                    TileImage(
                        file_path=str(resolved_path),
                        file_name=resolved_path.name,
                        file_size=file_size,
                        dimensions=dimensions,
                        file_mtime=file_mtime,
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
                )

        features_list = self._feature_extractor.extract_batch(
            path_strings,
            preprocess_workers=self._perf.adaptive_preprocess_workers(
                _max_file_bytes(items)
            ),
        )
        extract_timings = self._feature_extractor.last_timings
        timer.timings.record("preprocessing", extract_timings.preprocessing * len(items))
        timer.timings.record("dinov2", extract_timings.dinov2 * len(items))
        timer.timings.record("descriptors", extract_timings.descriptors * len(items))

        db_ids: List[int] = []
        vectors: List[List[float]] = []

        with timer.measure("database"):
            for tile, features in zip(tiles, features_list):
                tile.features = features
                db_id = self._repo.add(tile)
                tile.id = db_id
                tile.embedding_id = db_id
                db_ids.append(db_id)
                vectors.append(features.embedding.tolist())

        for item in items:
            generate_thumbnail(item.path, self._thumbnail_dir)

        with timer.measure("faiss"):
            self._index.update_vectors(db_ids, vectors, persist=persist)

        with timer.measure("database"):
            for tile in tiles:
                tile.is_indexed = True
                self._repo.add(tile)

        timer.log_summary(log=logger)

    def scan_and_index_directory(
        self,
        directory_path: Path,
        progress_callback: Optional[Callable[[int, int, str, float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        pause_event: Optional[threading.Event] = None,
        force: bool = False,
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
            force: If True, re-embed every file regardless of whether its
                content hash matches what's already indexed (Task D:
                Settings' "Rebuild FAISS Index" action — useful if the
                FAISS index file itself was lost/corrupted while SQLite
                metadata is still intact, since vectors aren't otherwise
                recoverable without re-running the embedding model).
                Every file is counted as "modified" in the returned
                ScanResult, not "new", since it was already known.

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
        pending_batch: List[_PendingIndexItem] = []
        indexed_since_checkpoint = 0

        def _flush_pending_batch() -> None:
            nonlocal new_count, modified_count, failed_count, indexed_since_checkpoint
            if not pending_batch:
                return

            batch_size = len(pending_batch)
            try:
                self._index_file_batch(pending_batch, persist=False)
                for item in pending_batch:
                    if item.was_previously_indexed:
                        modified_count += 1
                    else:
                        new_count += 1
                indexed_since_checkpoint += batch_size
            except Exception as e:
                failed_count += batch_size
                logger.error(
                    "Batch indexing failed for %d file(s): %s",
                    batch_size,
                    e,
                )
            finally:
                pending_batch.clear()

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
                _flush_pending_batch()
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

                if _is_unchanged_indexed_file(
                    resolved_path,
                    existing_record,
                    force=force,
                ):
                    logger.debug(f"Skipping unchanged tile: {file_name}")
                    skipped_count += 1
                    continue

                pending_batch.append(
                    _PendingIndexItem(
                        path=resolved_path,
                        was_previously_indexed=was_previously_indexed,
                    )
                )

                if len(pending_batch) >= self._pending_batch_limit(pending_batch):
                    _flush_pending_batch()

            except Exception as e:
                failed_count += 1
                logger.error(f"Error indexing file during folder scan: {resolved_path}. Error: {e}")

            if (
                indexed_since_checkpoint > 0
                and processed_count % self._perf.checkpoint_interval == 0
            ):
                self._index.save_index()
                indexed_since_checkpoint = 0

        _flush_pending_batch()

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
