"""
Drop nested indexed folders so a rebuild does not re-embed the same tiles.

Indexed-folder rows often include both a parent (e.g. ``.../Tiles``) and a
child (``.../Tiles/New``). ``scan_and_index_directory`` uses ``rglob``, so
scanning the parent already covers the child — looping both with
``force=True`` re-embeds the nested files a second time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Sequence

logger = logging.getLogger("tilevision.core.use_cases.index_folder_dedupe")


def drop_nested_indexed_folders(folder_paths: Sequence[str]) -> List[str]:
    """
    Keep only outermost folders from an indexed-folder list.

    Parents are preferred over children. Order of survivors follows the
    shortest-path-first discovery order (stable for a given set).
    """
    if not folder_paths:
        return []

    resolved: List[Path] = []
    seen: set[str] = set()
    for raw in folder_paths:
        try:
            path = Path(raw).expanduser().resolve()
        except OSError:
            path = Path(raw).expanduser().absolute()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path)

    # Shorter paths first so parents are kept before nested children.
    resolved.sort(key=lambda p: (len(str(p)), str(p).lower()))

    kept: List[Path] = []
    for path in resolved:
        parent_hit = next((parent for parent in kept if _is_under(path, parent)), None)
        if parent_hit is not None:
            logger.info(
                "Skipping nested indexed folder %s (already covered by %s)",
                path,
                parent_hit,
            )
            continue
        kept.append(path)

    return [str(p) for p in kept]


def _is_under(path: Path, parent: Path) -> bool:
    """True when ``path`` is ``parent`` or a descendant of ``parent``."""
    if path == parent:
        return True
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def merge_processed_paths(
    processed: set[str],
    paths: Iterable[str | Path],
) -> None:
    """Add resolved path strings to the cross-folder processed set."""
    for raw in paths:
        try:
            processed.add(str(Path(raw).resolve()))
        except OSError:
            processed.add(str(Path(raw).absolute()))
