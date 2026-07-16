"""
Indexing performance configuration for large tile catalogs.

Tunable via AppSettings (config.json) for batch size, decode limits,
and adaptive thresholds when indexing 70–200 MB production images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.settings import AppSettings


@dataclass(frozen=True, slots=True)
class IndexingPerformanceConfig:
    """Runtime tuning knobs for folder indexing throughput and memory."""

    batch_size: int = 12
    checkpoint_interval: int = 25
    max_decode_edge: int = 2048
    preprocess_workers: int = 4
    large_file_mb: int = 10
    huge_file_mb: int = 50

    def adaptive_batch_size(self, max_file_bytes: int) -> int:
        """Shrink batches when pending files are very large on disk."""
        mb = max_file_bytes / (1024 * 1024)
        if mb >= self.huge_file_mb:
            return min(2, self.batch_size)
        if mb >= self.large_file_mb:
            return min(4, self.batch_size)
        return self.batch_size

    def adaptive_preprocess_workers(self, max_file_bytes: int) -> int:
        """Reduce parallel decode pressure for huge source files."""
        mb = max_file_bytes / (1024 * 1024)
        if mb >= self.huge_file_mb:
            return 1
        if mb >= self.large_file_mb:
            return min(2, self.preprocess_workers)
        return self.preprocess_workers

    @classmethod
    def from_settings(
        cls,
        settings: AppSettings,
        *,
        use_gpu: bool = False,
    ) -> IndexingPerformanceConfig:
        batch_size = max(1, int(settings.index_batch_size))
        if use_gpu:
            batch_size = max(batch_size, int(settings.gpu_index_batch_size))

        return cls(
            batch_size=batch_size,
            checkpoint_interval=max(1, int(settings.index_checkpoint_interval)),
            max_decode_edge=max(512, int(settings.max_decode_edge)),
            preprocess_workers=max(1, int(settings.preprocess_workers)),
            large_file_mb=max(1, int(settings.large_file_mb_threshold)),
            huge_file_mb=max(1, int(settings.huge_file_mb_threshold)),
        )
