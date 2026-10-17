"""
Update / rebuild search index worker for TileVision AI.

Default mode is incremental ("Update Search Index"): only stale DINOv2
features, new/changed files, and missing CLIP description-search vectors
are processed. Full rebuild (force=True) re-embeds everything.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional

from PySide6.QtCore import QThread, Signal

from src.core.use_cases.index_folder_dedupe import drop_nested_indexed_folders
from src.core.use_cases.index_images import IndexImagesUseCase

logger = logging.getLogger("tilevision.presentation.workers.rebuild_index_worker")


@dataclass(slots=True)
class RebuildSummary:
    """Aggregate outcome across all folders in one update/rebuild run."""

    reindexed: int = 0
    clip_backfill: int = 0
    color_backfill: int = 0
    skipped: int = 0
    failed: int = 0
    folders: int = 0
    cancelled: bool = False
    force: bool = False

    def format_message(self) -> str:
        parts = [
            f"{self.reindexed} re-indexed",
            f"{self.clip_backfill} description-search vectors added",
            f"{self.skipped} already up to date",
        ]
        if self.color_backfill:
            parts.insert(2, f"{self.color_backfill} colours detected")
        if self.failed:
            parts.append(f"{self.failed} failed")
        return ", ".join(parts)


class RebuildIndexWorker(QThread):
    """Background worker that updates or force-rebuilds every known folder."""

    # (processed_files, total_files, current_filename, eta_seconds)
    progress_updated = Signal(int, int, str, float)

    # RebuildSummary
    rebuild_finished = Signal(object)

    rebuild_failed = Signal(str)

    def __init__(
        self,
        use_case: IndexImagesUseCase,
        folder_paths: List[str],
        *,
        force: bool = False,
    ) -> None:
        """
        Args:
            use_case: Fully configured IndexImagesUseCase.
            folder_paths: Every folder listed in IIndexedFolderRepository.
            force: When True, full re-embed (legacy Rebuild). When False,
                incremental Update Search Index.
        """
        super().__init__()
        self._use_case = use_case
        # Drop nested folders (parent rglob already covers children).
        self._folder_paths = drop_nested_indexed_folders(folder_paths)
        self._force = force
        self._cancel_event = threading.Event()
        self.last_summary: Optional[RebuildSummary] = None

    def request_cancel(self) -> None:
        """Ask the scan loop to stop at the next cooperative cancel check."""
        logger.warning("Rebuild cancel requested by user")
        self._cancel_event.set()

    def run(self) -> None:
        summary = RebuildSummary(
            folders=len(self._folder_paths),
            force=self._force,
        )
        # Cross-folder guard: even after nested drop, never embed the same
        # resolved path twice in one run.
        processed_paths: set[str] = set()

        try:
            for folder_path in self._folder_paths:
                if self._cancel_event.is_set():
                    break

                def on_progress(
                    processed: int,
                    total: int,
                    current_filename: str,
                    eta_seconds: float,
                ) -> None:
                    self.progress_updated.emit(
                        processed,
                        total,
                        current_filename,
                        eta_seconds,
                    )

                result = self._use_case.scan_and_index_directory(
                    folder_path,
                    force=self._force,
                    progress_callback=on_progress,
                    cancel_event=self._cancel_event,
                    processed_paths=processed_paths,
                )
                summary.reindexed += result.indexed_count
                summary.clip_backfill += int(result.clip_backfill_count or 0)
                summary.color_backfill += int(result.color_backfill_count or 0)
                summary.skipped += result.skipped_count
                summary.failed += result.failed_count

            summary.cancelled = self._cancel_event.is_set()
            self.last_summary = summary

            if summary.cancelled:
                logger.warning(
                    "Search index update cancelled: %s",
                    summary.format_message(),
                )
            else:
                mode = "full rebuild" if self._force else "update"
                logger.info(
                    "Search index %s complete: %s across %s folder(s).",
                    mode,
                    summary.format_message(),
                    summary.folders,
                )
            self.rebuild_finished.emit(summary)
        except Exception as e:
            logger.error(f"Search index update failed: {e}")
            self.rebuild_failed.emit(str(e))
