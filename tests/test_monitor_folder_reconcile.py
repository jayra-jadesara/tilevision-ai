"""Tests for reconcile-on-start catch-up indexing."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.use_cases.monitor_folder import (
    FolderMonitorController,
    TileImageEventHandler,
    is_watchdog_available,
)


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(40, 120, 80)).save(path, format="PNG")


def _wait_until(predicate, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"Condition not met within {timeout}s")


def test_reconcile_existing_file_indexes_without_fs_event(tmp_path: Path) -> None:
    use_case = MagicMock()
    use_case.index_changed_file.return_value = 7
    events: list = []

    handler = TileImageEventHandler(
        indexing_use_case=use_case,
        on_file_indexed_callback=lambda p, a, s, m: events.append((p, a, s, m)),
        settle_delay_seconds=5.0,  # would be slow if settle path were used
        debounce_seconds=5.0,
    )

    image = tmp_path / "preexisting.png"
    _write_png(image)

    handler.reconcile_existing_file(str(image))

    use_case.index_changed_file.assert_called_once()
    assert events[-1][1] == "indexed"
    # Must not go through settle/debounce timers.
    assert events[-1][0].endswith("preexisting.png")


def test_reconcile_skips_unchanged_files(tmp_path: Path) -> None:
    use_case = MagicMock()
    use_case.index_changed_file.return_value = None
    events: list = []
    handler = TileImageEventHandler(
        indexing_use_case=use_case,
        on_file_indexed_callback=lambda p, a, s, m: events.append((p, a, s, m)),
    )
    image = tmp_path / "same.png"
    _write_png(image)

    handler.reconcile_existing_file(str(image))

    assert events[-1][1] == "skipped"


@pytest.mark.skipif(not is_watchdog_available(), reason="watchdog not installed")
def test_start_monitoring_reconciles_preexisting_images(tmp_path: Path) -> None:
    """Files already in the folder are indexed on start without an FS event."""
    watch_dir = tmp_path / "watched"
    nested = watch_dir / "sub"
    image_a = watch_dir / "a.png"
    image_b = nested / "b.png"
    _write_png(image_a)
    _write_png(image_b)

    use_case = MagicMock()
    indexed_paths: list[str] = []

    def _index(path):
        indexed_paths.append(str(Path(path).resolve()))
        return 100 + len(indexed_paths)

    use_case.index_changed_file.side_effect = _index

    events: list = []
    controller = FolderMonitorController(
        indexing_use_case=use_case,
        on_file_indexed_callback=lambda p, a, s, m: events.append((p, a, s)),
    )

    try:
        controller.start_monitoring([str(watch_dir)])
        assert controller.is_running

        _wait_until(lambda: len(indexed_paths) >= 2, timeout=5.0)

        resolved = {str(Path(p).resolve()) for p in indexed_paths}
        assert str(image_a.resolve()) in resolved
        assert str(image_b.resolve()) in resolved
        indexed_actions = [e for e in events if e[1] == "indexed"]
        assert len(indexed_actions) >= 2
    finally:
        controller.stop_monitoring()
