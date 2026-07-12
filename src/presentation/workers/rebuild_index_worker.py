"""
Rebuild index worker module for TileVision AI (Task D: Settings — Rebuild
FAISS Index).

Runs a forced full re-embed of every previously-indexed folder on a
background QThread. This is the practical meaning of "rebuild" in this
architecture: since raw embedding vectors aren't persisted anywhere except
inside the FAISS index file itself, the only way to truly rebuild it (e.g.
after the index file was lost/corrupted while SQLite metadata survived) is
to re-run the embedding model against every known file.
"""

import logging
from typing import List

from PySide6.QtCore import QThread, Signal

from src.core.use_cases.index_images import IndexImagesUseCase

logger = logging.getLogger("tilevision.presentation.workers.rebuild_index_worker")


class RebuildIndexWorker(QThread):
    """Background worker that force-reindexes every known folder."""

    # (processed_folders, total_folders, current_folder_name)
    progress_updated = Signal(int, int, str)

    # (total_files_reembedded, total_failed)
    rebuild_finished = Signal(int, int)

    rebuild_failed = Signal(str)

    def __init__(self, use_case: IndexImagesUseCase, folder_paths: List[str]) -> None:
        """
        Args:
            use_case: Fully configured IndexImagesUseCase.
            folder_paths: Every folder to force-reindex (typically every
                row from IIndexedFolderRepository.get_all_folders()).
        """
        super().__init__()
        self._use_case = use_case
        self._folder_paths = folder_paths

    def run(self) -> None:
        total_reembedded = 0
        total_failed = 0
        total = len(self._folder_paths)

        try:
            for i, folder_path in enumerate(self._folder_paths):
                self.progress_updated.emit(i, total, folder_path)
                result = self._use_case.scan_and_index_directory(folder_path, force=True)
                total_reembedded += result.indexed_count
                total_failed += result.failed_count

            self.progress_updated.emit(total, total, "Done")
            logger.info(
                f"FAISS rebuild complete: {total_reembedded} file(s) re-embedded, "
                f"{total_failed} failed, across {total} folder(s)."
            )
            self.rebuild_finished.emit(total_reembedded, total_failed)
        except Exception as e:
            logger.error(f"FAISS rebuild failed: {e}")
            self.rebuild_failed.emit(str(e))
