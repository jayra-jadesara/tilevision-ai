"""
Image indexing use case module for TileVision AI.

Coordinates scanning file directories, validating image structures, generating cached
thumbnails, generating embeddings via the AI model, and indexing into the FAISS database.
Supports incremental indexing, perceptual hashing, SHA-256 matching, and cooperative thread execution.
"""

import copy
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.config.indexing_performance import IndexingPerformanceConfig
from src.core.models import TileImage, ScanResult, IndexedFolderState
from src.core.use_cases.index_eta import typical_work_seconds
from src.core.use_cases.index_skip import is_unchanged_indexed_file
from src.core.use_cases.index_timeout import (
    IndexFileTimeoutError,
    IndexStallAbort,
    file_index_timeout_s,
    inference_lock_is_held,
    run_with_timeout,
    stall_abort_message,
)
from src.core.use_cases.index_work_classify import IndexWorkKind, classify_index_work
from src.data.repository_interface import IImageRepository, IIndexedFolderRepository
from src.ai.feature_extractor import FeatureExtractor
from src.ai.inference_guard import wait_while_search_priority
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
    work: str = IndexWorkKind.FULL.value
    db_id: Optional[int] = None


def _max_file_bytes(items: List[_PendingIndexItem]) -> int:
    return max(item.path.stat().st_size for item in items)


def _index_content_key(sha256_hash: str, item_index: int) -> str:
    """Stable dedupe key for byte-identical catalogue files in one batch."""
    return sha256_hash if sha256_hash else f"__unique:{item_index}"


def _clone_index_vectors(features, aux: List) -> Tuple:
    """Copy vectors so duplicate paths do not share mutable numpy buffers."""
    import numpy as np

    cloned_features = copy.deepcopy(features)
    cloned_aux = [np.asarray(vec, dtype=np.float32).copy() for vec in aux]
    return cloned_features, cloned_aux


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
        clip_embedder=None,
        clip_index: Optional[FaissIndexManager] = None,
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
            clip_embedder / clip_index: Optional CLIP dual-index writers
                for description search. When omitted, indexing is DINOv2-only.
        """
        self._repo = image_repository
        self._feature_extractor = feature_extractor
        self._index = vector_index
        self._thumbnail_dir = Path(thumbnail_dir)
        self._supported_extensions = set(SUPPORTED_IMAGE_EXTENSIONS)
        self._folder_repo = folder_repository
        self._perf = performance or IndexingPerformanceConfig()
        self._clip_embedder = clip_embedder
        self._clip_index = clip_index

        # Ensure thumbnail directory exists
        self._thumbnail_dir.mkdir(parents=True, exist_ok=True)

    def _embed_clip_vector(self, image_path: Path) -> Optional[List[float]]:
        """Return a CLIP vector for ``image_path``, or None when CLIP is unavailable."""
        if self._clip_embedder is None or self._clip_index is None:
            return None
        # Skip quietly when CLIP is not on disk yet and has never been loaded —
        # customers who never use description search must not pay a Hub download
        # during every catalogue index.
        try:
            from src.ai.clip_model_paths import clip_model_available_locally

            ready = bool(getattr(self._clip_embedder, "is_loaded", False)) or (
                clip_model_available_locally()
            )
        except Exception:
            ready = bool(getattr(self._clip_embedder, "is_loaded", False))
        if not ready:
            return None
        try:
            if getattr(self._clip_index, "_index", None) is None:
                self._clip_index.load_index()
            return self._clip_embedder.embed_image(image_path)
        except Exception as exc:
            logger.warning(
                "CLIP embed skipped for %s: %s",
                image_path.name,
                exc,
            )
            return None

    def _index_clip_vector(self, db_id: int, image_path: Path, *, persist: bool) -> None:
        """Best-effort second embedding for description search (never fails DINOv2)."""
        vec = self._embed_clip_vector(image_path)
        if vec is None or self._clip_index is None:
            return
        try:
            self._clip_index.update_vectors([db_id], [vec], persist=persist)
        except Exception as exc:
            logger.warning(
                "CLIP index write skipped for id=%s (%s): %s",
                db_id,
                image_path.name,
                exc,
            )

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
        wait_while_search_priority()
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
        features, aux_vectors = self._feature_extractor.extract_index_vectors(
            str(resolved_path)
        )
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
            # Aux FAISS vectors (same id): texture-panel for marketing sheets
            # and/or center-50% for large tiles so customer crops still hit.
            logger.info(
                f"Indexing vector into FAISS for ID {db_id}: {file_name}"
            )

            faiss_ids = [db_id]
            faiss_vectors = [features.embedding]
            for aux in aux_vectors:
                faiss_ids.append(db_id)
                faiss_vectors.append(aux)

            with timer.measure("faiss"):
                self._index.update_vectors(
                    faiss_ids,
                    faiss_vectors,
                    persist=persist,
                )

            # Optional CLIP dual-index (description search) — same db_id.
            with timer.measure("clip"):
                self._index_clip_vector(db_id, resolved_path, persist=persist)

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
        if is_unchanged_indexed_file(resolved_path, existing_record, force=False):
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
        Index multiple files from a folder scan.

        Uses the same ``extract_index_vectors()`` path as ``index_single_file``
        (``prepare_index_primary`` panel isolation + aux FAISS vectors). The
        old ``extract_batch()`` + ``ImagePreprocessor.preprocess()`` path wrote
        full-sheet descriptors to SQLite and caused PGYS2319-style hybrid drift.
        """
        if not items:
            return

        wait_while_search_priority()
        timer = PipelineTimer("INDEX BATCH TIMING")
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

        features_list: List = []
        aux_by_index: List[List] = []

        rep_paths: List[str] = []
        rep_keys_order: List[str] = []
        seen_content_keys: set = set()
        first_row_for_key: dict = {}
        for i, (item, tile) in enumerate(zip(items, tiles)):
            key = _index_content_key(tile.sha256_hash, i)
            if key not in seen_content_keys:
                seen_content_keys.add(key)
                rep_keys_order.append(key)
                rep_paths.append(str(item.path))
            if key not in first_row_for_key:
                first_row_for_key[key] = i

        reused_identical = len(items) - len(rep_paths)

        with timer.measure("feature_extract"):
            batch_fn = getattr(
                self._feature_extractor, "extract_index_vectors_batch", None
            )
            if batch_fn is not None and len(rep_paths) > 1:
                extracted_reps = batch_fn(rep_paths)
            else:
                extracted_reps = [
                    self._feature_extractor.extract_index_vectors(path)
                    for path in rep_paths
                ]
            extracted_by_key = {
                key: pair for key, pair in zip(rep_keys_order, extracted_reps)
            }
            for i, tile in enumerate(tiles):
                key = _index_content_key(tile.sha256_hash, i)
                features, aux = extracted_by_key[key]
                if i != first_row_for_key[key]:
                    features, aux = _clone_index_vectors(features, aux)
                features_list.append(features)
                aux_by_index.append(aux)

        if reused_identical > 0:
            logger.info(
                "reused embeddings for %d identical files",
                reused_identical,
            )

        timings = self._feature_extractor.last_timings
        embed_count = len(rep_paths)
        # last_timings are already per-file averages from the batch helper.
        timer.timings.record("preprocessing", timings.preprocessing * embed_count)
        timer.timings.record("dinov2", timings.dinov2 * embed_count)
        timer.timings.record("descriptors", timings.descriptors * embed_count)

        db_ids: List[int] = []
        faiss_ids: List[int] = []
        faiss_vectors: List[List[float]] = []

        with timer.measure("database"):
            for tile, features in zip(tiles, features_list):
                tile.features = features
                db_id = self._repo.add(tile)
                tile.id = db_id
                tile.embedding_id = db_id
                db_ids.append(db_id)

        for item in items:
            generate_thumbnail(item.path, self._thumbnail_dir)

        for db_id, features, aux in zip(db_ids, features_list, aux_by_index):
            faiss_ids.append(db_id)
            faiss_vectors.append(features.embedding)
            for aux_vec in aux:
                faiss_ids.append(db_id)
                faiss_vectors.append(aux_vec)

        with timer.measure("faiss"):
            self._index.update_vectors(faiss_ids, faiss_vectors, persist=persist)

        with timer.measure("clip"):
            # Add all CLIP vectors first; persist once per batch (not per tile).
            # Per-tile save rewrote tiles_clip.index + sidecar N times per batch
            # (real Windows log: 12 full rewrites in ~2s for one batch of 12).
            clip_by_content: dict = {}
            for db_id, item, tile in zip(db_ids, items, tiles):
                clip_key = (
                    tile.sha256_hash if tile.sha256_hash else f"__id:{db_id}"
                )
                cached = clip_by_content.get(clip_key)
                if cached is not None:
                    if self._clip_index is not None:
                        self._clip_index.update_vectors(
                            [db_id], [cached], persist=False
                        )
                    continue
                vec = self._embed_clip_vector(item.path)
                if vec is None:
                    continue
                clip_by_content[clip_key] = vec
                if self._clip_index is not None:
                    self._clip_index.update_vectors(
                        [db_id], [vec], persist=False
                    )
            if persist and self._clip_index is not None:
                try:
                    self._clip_index.save_index()
                except Exception as exc:
                    logger.warning("CLIP index batch save skipped: %s", exc)

        with timer.measure("database"):
            for tile in tiles:
                tile.is_indexed = True
                self._repo.add(tile)

        timer.log_summary(log=logger)

    def _clip_backfill_batch(
        self,
        items: List[_PendingIndexItem],
        *,
        persist: bool = True,
    ) -> int:
        """
        Write CLIP vectors only for tiles whose DINOv2 features are already current.

        ~55× cheaper than a full re-embed on CPU (CLIP ~0.2 s vs DINOv2 ~11 s).
        """
        if not items or self._clip_embedder is None or self._clip_index is None:
            return 0
        done = 0
        for item in items:
            db_id = item.db_id
            if db_id is None:
                existing = self._repo.get_by_path(str(item.path.resolve()))
                db_id = existing.id if existing is not None else None
            if db_id is None:
                logger.warning(
                    "CLIP backfill skipped (no db id) for %s", item.path.name
                )
                continue
            try:
                self._index_clip_vector(int(db_id), item.path, persist=False)
                done += 1
            except Exception as exc:
                logger.warning(
                    "CLIP backfill failed for %s: %s", item.path.name, exc
                )
        if persist and done > 0:
            try:
                self._clip_index.save_index()
            except Exception as exc:
                logger.warning("CLIP backfill save skipped: %s", exc)
        return done

    def scan_and_index_directory(
        self,
        directory_path: Path,
        progress_callback: Optional[Callable[[int, int, str, float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        pause_event: Optional[threading.Event] = None,
        force: bool = False,
        processed_paths: Optional[set] = None,
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
            progress_callback: Optional callback receiving
                (completed_count, total_count, status_name, eta_seconds).
                ``completed_count`` is files finished (not merely queued).
                ``eta_seconds`` is -1 until a real batch timing sample exists
                (UI must not invent a seconds estimate from queue time).
            cancel_event: Threading event to stop the scan loop.
            pause_event: Threading event to temporarily pause the scan loop.
            force: If True, re-embed every file (full rebuild). When False
                (Update Search Index), only stale / missing / CLIP-gap tiles
                are processed.
            processed_paths: Optional set of resolved path strings already
                handled earlier in this rebuild run (nested-folder guard).
                Mutated in place as files finish.

        Returns:
            A ScanResult with the new/modified/deleted/skipped/clip_backfill
            breakdown.
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
        clip_backfill_count = 0
        is_completed = True
        cross_run_seen: set = processed_paths if processed_paths is not None else set()

        self._index.load_index()
        clip_enabled = self._clip_embedder is not None and self._clip_index is not None
        clip_ids_present: Optional[set] = None
        if clip_enabled and not force:
            try:
                if getattr(self._clip_index, "_index", None) is None:
                    self._clip_index.load_index()
                clip_ids_present = self._clip_index.known_ids()
            except Exception as exc:
                logger.warning("Could not load CLIP id set for backfill: %s", exc)
                clip_ids_present = set()

        # perf_counter: Windows time.time() can round sub-15ms scans to 0 and
        # zero-out ETA during fast FakeEmbedder / tiny-folder tests.
        start_time = time.perf_counter()
        # Files finished (skipped / embedded / failed) — NOT files merely queued.
        # Emitting progress on queue made the UI jump to ~11/38 in <1s then sit
        # idle for minutes while batch 1 embedded (real Windows log evidence).
        completed_count = 0

        # Track every path we actually see on disk this scan, so we can
        # detect deletions afterward (any previously-indexed tile under
        # this folder whose path isn't in this set was removed from disk).
        paths_seen_on_disk: set = set()
        pending_batch: List[_PendingIndexItem] = []
        pending_clip_batch: List[_PendingIndexItem] = []

        # Per-batch wall-clock (seconds per image) for files that re-embed.
        # First flush often includes one-time model warmup (DINOv2/CLIP load).
        # Never feed that warm-up batch alone into ETA — wait for a second
        # sample (real log: 59 s/image first vs ~12.6 s steady → "~46 min"
        # when true total was ~14 min).
        batch_timings_log: List[float] = []
        warmup_batches_skip = 1
        recent_batch_window = 4
        # First batch is smaller so the first real timing sample arrives sooner
        # (~30–40s on CPU) instead of waiting on a full batch of 12 (~3 min).
        first_batch_max = 4

        def _typical_work_seconds() -> float:
            return typical_work_seconds(
                batch_timings_log,
                warmup_batches_skip=warmup_batches_skip,
                recent_batch_window=recent_batch_window,
            )

        def _flush_threshold(items: List[_PendingIndexItem]) -> int:
            base = self._pending_batch_limit(items)
            if not batch_timings_log:
                return max(1, min(first_batch_max, base))
            return base

        def _eta_seconds(completed: int) -> float:
            """
            Seconds remaining, or -1.0 when no real timing sample exists yet.

            Never invent ETA from queue-scan elapsed/processed — that produced
            the false "~10s remaining" while the first batch still had minutes left.
            """
            remaining_files = total_files - completed
            if remaining_files <= 0:
                return 0.0
            typical = _typical_work_seconds()
            if typical <= 0.0:
                return -1.0
            decided = skipped_count + new_count + modified_count + failed_count
            if decided > 0:
                work_rate = 1.0 - (skipped_count / decided)
            else:
                work_rate = 1.0
            return typical * remaining_files * max(0.0, min(1.0, work_rate))

        def _emit_progress(name: str, *, eta: Optional[float] = None) -> None:
            if not progress_callback:
                return
            if eta is None:
                eta = _eta_seconds(completed_count)
            progress_callback(completed_count, total_files, name, float(eta))

        def _record_success(item: _PendingIndexItem) -> None:
            nonlocal new_count, modified_count
            if item.was_previously_indexed:
                modified_count += 1
            else:
                new_count += 1

        def _index_one_with_timeout(item: _PendingIndexItem) -> None:
            """Embed one file with a hard wall-clock timeout."""
            label = item.path.name
            timeout_s = file_index_timeout_s()
            t0 = time.perf_counter()
            logger.info(
                "INDEX FILE START: %s timeout=%.0fs force=%s",
                label,
                timeout_s,
                force,
            )
            try:
                run_with_timeout(
                    lambda: self._index_file_batch([item], persist=True),
                    timeout_s=timeout_s,
                    label=label,
                )
                logger.info(
                    "INDEX FILE END: %s ok elapsed=%.2fs",
                    label,
                    time.perf_counter() - t0,
                )
            except IndexFileTimeoutError:
                logger.error(
                    "INDEX FILE END: %s FAILED timeout after %.2fs",
                    label,
                    time.perf_counter() - t0,
                )
                raise
            except Exception:
                logger.exception(
                    "INDEX FILE END: %s FAILED elapsed=%.2fs",
                    label,
                    time.perf_counter() - t0,
                )
                raise

        def _flush_per_file(items: List[_PendingIndexItem]) -> Tuple[int, int]:
            """
            Isolate a bad file after a batch failure.

            On per-file timeout we abort the whole scan: the hung worker still
            holds ``_INFERENCE_LOCK``, so continuing would only time out again.
            """
            nonlocal failed_count
            timed_out = 0
            other_failed = 0
            for item in items:
                if cancel_event and cancel_event.is_set():
                    break
                try:
                    _index_one_with_timeout(item)
                    _record_success(item)
                except IndexFileTimeoutError:
                    timed_out += 1
                    failed_count += 1
                    raise IndexStallAbort(
                        stall_abort_message(item.path.name, file_index_timeout_s())
                    )
                except Exception as e:
                    other_failed += 1
                    failed_count += 1
                    logger.error(
                        "Per-file indexing failed for %s: %s",
                        item.path.name,
                        e,
                    )
            return timed_out, other_failed

        def _flush_pending_batch() -> None:
            nonlocal new_count, modified_count, failed_count, is_completed, completed_count
            if not pending_batch:
                return

            items = list(pending_batch)
            pending_batch.clear()
            batch_size = len(items)
            flush_t0 = time.perf_counter()
            names = ",".join(item.path.name for item in items[:6])
            if batch_size > 6:
                names += f",…(+{batch_size - 6})"
            per_file_timeout = file_index_timeout_s()
            # Whole-batch budget scales with size so normal batched embeds
            # are not cut short; hung forwards still return at this wall clock.
            batch_timeout = per_file_timeout * max(1, batch_size)
            is_first_batch = not batch_timings_log
            if is_first_batch:
                _emit_progress(
                    f"Embedding first batch of {batch_size} images…",
                    eta=-1.0,
                )
            else:
                _emit_progress(f"Embedding {batch_size} images…")
            logger.info(
                "INDEX BATCH START: n=%d timeout=%.0fs files=[%s]",
                batch_size,
                batch_timeout,
                names,
            )
            timed_out = 0
            other_failed = 0
            used_fallback = False
            try:
                try:
                    run_with_timeout(
                        lambda: self._index_file_batch(items, persist=True),
                        timeout_s=batch_timeout,
                        label=f"batch[{batch_size}:{names}]",
                    )
                    for item in items:
                        _record_success(item)
                except IndexFileTimeoutError:
                    # Timed-out batch worker usually still holds the inference
                    # RLock — continuing (even per-file) would hang again.
                    lock_held = inference_lock_is_held()
                    if lock_held is not False:
                        failed_count += batch_size
                        timed_out = batch_size
                        raise IndexStallAbort(
                            stall_abort_message(
                                items[0].path.name if items else "batch",
                                batch_timeout,
                            )
                        )
                    logger.warning(
                        "INDEX BATCH TIMEOUT but inference lock free — "
                        "falling back to per-file isolation"
                    )
                    used_fallback = True
                    timed_out, other_failed = _flush_per_file(items)
                except IndexStallAbort:
                    raise
                except Exception as e:
                    # Non-timeout batch failure: isolate the bad file(s).
                    logger.warning(
                        "INDEX BATCH failed (%s); falling back to per-file",
                        e,
                    )
                    used_fallback = True
                    timed_out, other_failed = _flush_per_file(items)

                done = batch_size - timed_out - other_failed
                # When we fell back, successes were already counted in
                # _record_success / failures in failed_count.
                if used_fallback:
                    done = max(0, batch_size - timed_out - other_failed)
                finished_this_flush = done + timed_out + other_failed
                completed_count += finished_this_flush
                per_image = (time.perf_counter() - flush_t0) / max(1, done or batch_size)
                per_image = max(per_image, 1e-6)
                if done > 0:
                    batch_timings_log.append(per_image)
                    if len(batch_timings_log) > 32:
                        del batch_timings_log[: len(batch_timings_log) - 32]
                typical = _typical_work_seconds()
                logger.info(
                    "INDEX BATCH END: n=%d ok=%d timeout=%d failed=%d "
                    "fallback=%s per_image=%.3fs typical_steady=%.3fs "
                    "batch_times=%s",
                    batch_size,
                    done,
                    timed_out,
                    other_failed,
                    used_fallback,
                    per_image,
                    typical,
                    ",".join(f"{s:.3f}" for s in batch_timings_log[-8:]),
                )
                logger.info(
                    "INDEX TIMING per-file: batch_n=%d per_image=%.3fs "
                    "batch_index=%d typical_steady=%.3fs batch_times=%s",
                    max(1, done or batch_size),
                    per_image,
                    len(batch_timings_log),
                    typical,
                    ",".join(f"{s:.3f}" for s in batch_timings_log[-8:]),
                )
                for item in items:
                    cross_run_seen.add(str(item.path.resolve()))
                    if clip_ids_present is not None and item.db_id is None:
                        # After full embed, id is on the repo row.
                        rec = self._repo.get_by_path(str(item.path.resolve()))
                        if rec is not None and rec.id is not None:
                            clip_ids_present.add(int(rec.id))
                last_name = items[-1].path.name if items else "batch"
                typical_note = ""
                if typical > 0:
                    remaining = max(0, total_files - completed_count)
                    typical_note = f" (~{typical:.0f}s/image"
                    if remaining:
                        typical_note += f", ~{int(typical * remaining)}s left"
                    typical_note += ")"
                elif len(batch_timings_log) <= warmup_batches_skip:
                    typical_note = " (estimating…)"
                _emit_progress(f"{last_name}{typical_note}")
            except IndexStallAbort:
                is_completed = False
                if cancel_event is not None:
                    cancel_event.set()
                logger.error("INDEX STALL ABORT — stopping scan")
                raise

        def _flush_clip_batch() -> None:
            nonlocal clip_backfill_count, completed_count, failed_count
            if not pending_clip_batch:
                return
            items = list(pending_clip_batch)
            pending_clip_batch.clear()
            _emit_progress(
                f"Adding description-search vectors for {len(items)} images…",
                eta=_eta_seconds(completed_count),
            )
            try:
                done = self._clip_backfill_batch(items, persist=True)
                clip_backfill_count += done
                completed_count += len(items)
                for item in items:
                    cross_run_seen.add(str(item.path.resolve()))
                    if clip_ids_present is not None and item.db_id is not None:
                        clip_ids_present.add(int(item.db_id))
                last_name = items[-1].path.name if items else "clip"
                _emit_progress(f"{last_name} (description-search vectors)")
            except Exception as exc:
                failed_count += len(items)
                completed_count += len(items)
                logger.error("CLIP backfill batch failed: %s", exc)
                _emit_progress("description-search backfill failed")

        if progress_callback:
            progress_callback(
                0,
                total_files,
                "Scanning catalogue…",
                -1.0,
            )

        for file_path in all_files:
            # 0. Yield to drop-image Search so results can return.
            wait_while_search_priority()

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
                _flush_clip_batch()
                is_completed = False
                break

            resolved_path = file_path.resolve()
            file_name = resolved_path.name
            resolved_key = str(resolved_path)
            paths_seen_on_disk.add(resolved_key)

            logger.debug("INDEX SCAN FILE: %s", file_name)

            try:
                # Already handled by a parent folder earlier in this rebuild.
                if resolved_key in cross_run_seen:
                    logger.debug(
                        "Skipping already-processed path this run: %s", file_name
                    )
                    skipped_count += 1
                    completed_count += 1
                    _emit_progress(file_name)
                    continue

                existing_record = self._repo.get_by_path(resolved_key)
                was_previously_indexed = bool(
                    existing_record and existing_record.is_indexed
                )
                work = classify_index_work(
                    resolved_path,
                    existing_record,
                    force=force,
                    clip_ids_present=clip_ids_present,
                    clip_enabled=clip_enabled,
                )

                if work is IndexWorkKind.SKIP:
                    logger.debug(f"Skipping unchanged tile: {file_name}")
                    skipped_count += 1
                    completed_count += 1
                    cross_run_seen.add(resolved_key)
                    _emit_progress(file_name)
                    continue

                if work is IndexWorkKind.CLIP_ONLY:
                    pending_clip_batch.append(
                        _PendingIndexItem(
                            path=resolved_path,
                            was_previously_indexed=was_previously_indexed,
                            work=IndexWorkKind.CLIP_ONLY.value,
                            db_id=existing_record.id if existing_record else None,
                        )
                    )
                    if len(pending_clip_batch) >= _flush_threshold(pending_clip_batch):
                        _flush_clip_batch()
                    continue

                pending_batch.append(
                    _PendingIndexItem(
                        path=resolved_path,
                        was_previously_indexed=was_previously_indexed,
                        work=IndexWorkKind.FULL.value,
                        db_id=existing_record.id if existing_record else None,
                    )
                )

                if len(pending_batch) >= _flush_threshold(pending_batch):
                    _flush_pending_batch()

            except IndexStallAbort:
                is_completed = False
                raise
            except Exception as e:
                failed_count += 1
                completed_count += 1
                logger.error(f"Error indexing file during folder scan: {resolved_path}. Error: {e}")
                _emit_progress(file_name)

        try:
            _flush_pending_batch()
            _flush_clip_batch()
        except IndexStallAbort:
            is_completed = False
            raise

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
        if (new_count + modified_count + clip_backfill_count) > 0 or deleted_count > 0:
            self._index.save_index()
            if self._clip_index is not None:
                try:
                    self._clip_index.save_index()
                except Exception as exc:
                    logger.warning("CLIP index save skipped: %s", exc)
            logger.info(
                "Index directory complete. New: %s, Modified: %s, "
                "CLIP backfill: %s, Deleted: %s, Skipped: %s, Failed: %s",
                new_count,
                modified_count,
                clip_backfill_count,
                deleted_count,
                skipped_count,
                failed_count,
            )

        # Record this folder as indexed (Task 1) so the Index page can
        # restore its state on next startup without re-scanning. Only
        # recorded on a completed (non-cancelled) scan.
        if is_completed and self._folder_repo is not None:
            self._folder_repo.record_folder_indexed(str(root))

        elapsed_total = time.perf_counter() - start_time

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
            clip_backfill_count=clip_backfill_count,
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
