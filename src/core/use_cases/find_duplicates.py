"""
Duplicate detection use case module for TileVision AI (Feature 5).

Finds two kinds of duplicates across the indexed catalog:

  - Exact duplicates: tiles whose file content is byte-for-byte identical
    (same SHA-256 hash), typically the same image saved/copied under two
    different filenames or into two different folders.
  - Near duplicates: visually near-identical tiles that aren't byte-identical
    — e.g. the same photo re-saved at a different JPEG quality, resized, or
    with a watermark added — detected via Hamming distance between
    perceptual hashes (dHash).

Grouping uses a simple union-find so that A~B and B~C correctly merge into
a single {A, B, C} group even if A and C aren't within the threshold of
each other directly (transitive near-duplicate chains).
"""

import logging
from typing import Dict, List

from src.core.models import TileImage
from src.data.repository_interface import IImageRepository
from src.utils.image_utils import hamming_distance

logger = logging.getLogger("tilevision.core.use_cases.find_duplicates")

# Two perceptual hashes within this many differing bits (out of 64) are
# considered near-duplicates. Empirically, near-identical images (same
# photo, different compression/resize) typically land under ~10; visually
# distinct images are usually 20+.
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 8

# Pairwise near-duplicate comparison is O(n^2) in catalog size. For very
# large catalogs this use case still returns correct results but can take
# a while; this cap exists so a UI can warn the user or skip straight to
# exact-duplicate-only mode rather than freezing on a huge catalog.
NEAR_DUPLICATE_CATALOG_SIZE_WARNING_THRESHOLD = 20_000


class _UnionFind:
    """Minimal union-find (disjoint set) for clustering near-duplicate groups."""

    def __init__(self, items: List[int]) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: int) -> int:
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]  # path compression
            item = self._parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_b] = root_a

    def groups(self) -> Dict[int, List[int]]:
        result: Dict[int, List[int]] = {}
        for item in self._parent:
            root = self.find(item)
            result.setdefault(root, []).append(item)
        return result


class FindDuplicatesUseCase:
    """Use case to find exact and near-duplicate tiles in the catalog."""

    def __init__(self, image_repository: IImageRepository, vector_index=None) -> None:
        """
        Args:
            image_repository: Repository interface for SQLite tile access.
            vector_index: Optional FaissIndexManager. Required only for
                delete_duplicate() to also remove the tile's FAISS vector;
                kept optional so existing construction sites/tests that
                only need read-only duplicate detection keep working
                unchanged.
        """
        self._repo = image_repository
        self._vector_index = vector_index

    @staticmethod
    def similarity_percent(hash_a: str, hash_b: str) -> float:
        """
        Convert a Hamming distance between two perceptual hashes into a
        0-100 "duplicate %" for display (Task B: Duplicate Detection UI).

        Args:
            hash_a: First tile's perceptual hash.
            hash_b: Second tile's perceptual hash.

        Returns:
            100.0 for identical hashes, decreasing as the hashes diverge;
            0.0 if either hash is invalid/empty.
        """
        distance = hamming_distance(hash_a, hash_b)
        if distance < 0:
            return 0.0
        return max(0.0, 100.0 - (distance / 64.0) * 100.0)

    def delete_duplicate(self, tile: TileImage, delete_file_from_disk: bool = True) -> None:
        """
        Remove a duplicate tile: from disk (optional), FAISS, and SQLite.

        Args:
            tile: The TileImage to remove (as returned by
                find_exact_duplicates()/find_near_duplicates()).
            delete_file_from_disk: If True (default), also deletes the
                underlying image file from disk. Set False to only remove
                it from the catalog (e.g. the file already moved/was
                deleted externally) without touching the filesystem.

        Raises:
            RuntimeError: If any step of the removal fails. Partial
                cleanup (e.g. file deleted but DB row removal failed) is
                logged clearly so the user isn't left with a silent
                inconsistency.
        """
        import os

        if delete_file_from_disk:
            try:
                if os.path.exists(tile.file_path):
                    os.remove(tile.file_path)
                    logger.info(f"Deleted duplicate file from disk: {tile.file_path}")
            except OSError as e:
                logger.error(f"Failed to delete file from disk: {tile.file_path}: {e}")
                raise RuntimeError(f"Could not delete file from disk: {e}") from e

        if self._vector_index is not None and tile.embedding_id is not None:
            try:
                self._vector_index.remove_vectors([tile.embedding_id])
                self._vector_index.save_index()
            except Exception as e:
                logger.error(f"Failed to remove FAISS vector for tile {tile.id}: {e}")
                raise RuntimeError(f"Could not remove vector from search index: {e}") from e

        if tile.id is not None:
            try:
                self._repo.remove(tile.id)
                logger.info(f"Removed duplicate tile from catalog: {tile.file_path}")
            except Exception as e:
                logger.error(f"Failed to remove tile {tile.id} from database: {e}")
                raise RuntimeError(f"Could not remove tile from database: {e}") from e

    def find_exact_duplicates(self) -> List[List[TileImage]]:
        """
        Group tiles that are byte-for-byte identical (same SHA-256 hash).

        Returns:
            A list of duplicate groups (each a list of 2+ TileImage
            records sharing the same content hash). Tiles with no
            duplicates are omitted entirely.
        """
        all_tiles = self._repo.get_all()
        by_hash: Dict[str, List[TileImage]] = {}

        for tile in all_tiles:
            if not tile.sha256_hash:
                continue
            by_hash.setdefault(tile.sha256_hash, []).append(tile)

        groups = [group for group in by_hash.values() if len(group) > 1]
        logger.info(f"Exact duplicate scan: {len(groups)} group(s) found among {len(all_tiles)} tiles.")
        return groups

    def find_near_duplicates(
        self, threshold: int = DEFAULT_NEAR_DUPLICATE_THRESHOLD
    ) -> List[List[TileImage]]:
        """
        Group visually near-identical tiles via perceptual hash Hamming
        distance clustering. Uses union-find so transitively-close chains
        (A~B, B~C) merge into one group even if A and C individually
        exceed the threshold.

        Args:
            threshold: Maximum Hamming distance (0-64) between two
                perceptual hashes to consider them near-duplicates. Lower
                = stricter (fewer false positives, may miss some real
                near-dupes). Default is tuned for typical
                recompression/resize scenarios.

        Returns:
            A list of near-duplicate groups (each a list of 2+ TileImage
            records). Exact duplicates are also included here (Hamming
            distance 0), so callers wanting exact-only should use
            find_exact_duplicates() instead/additionally.
        """
        all_tiles = [t for t in self._repo.get_all() if t.perceptual_hash]

        if len(all_tiles) > NEAR_DUPLICATE_CATALOG_SIZE_WARNING_THRESHOLD:
            logger.warning(
                f"Near-duplicate scan on a large catalog ({len(all_tiles)} tiles) — "
                f"this is an O(n^2) comparison and may take a while."
            )

        ids = [t.id for t in all_tiles if t.id is not None]
        tile_by_id = {t.id: t for t in all_tiles if t.id is not None}
        uf = _UnionFind(ids)

        n = len(all_tiles)
        for i in range(n):
            tile_i = all_tiles[i]
            if tile_i.id is None:
                continue
            for j in range(i + 1, n):
                tile_j = all_tiles[j]
                if tile_j.id is None:
                    continue
                distance = hamming_distance(tile_i.perceptual_hash, tile_j.perceptual_hash)
                if 0 <= distance <= threshold:
                    uf.union(tile_i.id, tile_j.id)

        groups = [
            [tile_by_id[tid] for tid in group_ids]
            for group_ids in uf.groups().values()
            if len(group_ids) > 1
        ]
        logger.info(f"Near-duplicate scan: {len(groups)} group(s) found among {n} tiles.")
        return groups
