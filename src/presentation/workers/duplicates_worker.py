"""
Duplicate detection worker module for TileVision AI.

Runs exact/near-duplicate scanning (Feature 5) on a background QThread,
since near-duplicate detection is an O(n^2) comparison that can take a
noticeable amount of time on large catalogs.
"""

import logging
import time

from PySide6.QtCore import QThread, Signal

from src.core.use_cases.find_duplicates import FindDuplicatesUseCase

logger = logging.getLogger("tilevision.presentation.workers.duplicates_worker")


class DuplicatesWorker(QThread):
    """Background worker that runs a duplicate detection scan."""

    # Signal payload: (exact_groups, near_groups) — both List[List[TileImage]]
    scan_completed = Signal(list, list)
    scan_failed = Signal(str)

    def __init__(self, use_case: FindDuplicatesUseCase, include_near_duplicates: bool = True) -> None:
        """
        Args:
            use_case: Fully configured FindDuplicatesUseCase.
            include_near_duplicates: If False, only exact duplicates are
                scanned (much faster — O(n) vs O(n^2) — useful for very
                large catalogs where a full near-duplicate scan is slow).
        """
        super().__init__()
        self._use_case = use_case
        self._include_near_duplicates = include_near_duplicates

    def run(self) -> None:
        start = time.monotonic()
        try:
            exact_groups = self._use_case.find_exact_duplicates()
            near_groups = (
                self._use_case.find_near_duplicates() if self._include_near_duplicates else []
            )
            elapsed = time.monotonic() - start
            logger.info(
                f"Duplicate scan finished in {elapsed:.2f}s: "
                f"{len(exact_groups)} exact group(s), {len(near_groups)} near-duplicate group(s)."
            )
            self.scan_completed.emit(exact_groups, near_groups)
        except Exception as e:
            logger.error(f"Duplicate scan failed: {e}")
            self.scan_failed.emit(str(e))
