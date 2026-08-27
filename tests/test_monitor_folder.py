"""Tests for auto folder monitoring event handling."""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.use_cases.monitor_folder import TileImageEventHandler


def _write_png(path: Path) -> None:
    Image.new("RGB", (8, 8), color=(120, 80, 40)).save(path, format="PNG")


def _wait_for_events(
    events: list,
    *,
    timeout: float = 2.0,
    min_count: int = 1,
) -> None:
    """Wait until the monitor callback has recorded events (avoids fixed-sleep flakes)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(events) >= min_count:
            return
        time.sleep(0.02)
    raise AssertionError(
        f"Expected at least {min_count} monitor event(s) within {timeout}s; "
        f"got {len(events)}: {events!r}"
    )


@pytest.fixture()
def handler(tmp_path: Path):
    use_case = MagicMock()
    use_case.index_changed_file.return_value = 42
    use_case.remove_indexed_file.return_value = True
    events: list[tuple[str, str, bool, str]] = []

    def callback(path, action, success, message):
        events.append((path, action, success, message))

    event_handler = TileImageEventHandler(
        indexing_use_case=use_case,
        on_file_indexed_callback=callback,
        settle_delay_seconds=0.05,
        debounce_seconds=0.05,
    )
    return event_handler, use_case, events


def test_on_modified_schedules_index(tmp_path: Path, handler) -> None:
    event_handler, use_case, events = handler
    image = tmp_path / "tile.png"
    _write_png(image)

    class Event:
        is_directory = False
        src_path = str(image)

    with patch("src.core.use_cases.monitor_folder.validate_image", return_value=True):
        event_handler.on_modified(Event())
        _wait_for_events(events)

    use_case.index_changed_file.assert_called_once()
    assert events[-1][1] == "indexed"


def test_on_deleted_removes_from_index(tmp_path: Path, handler) -> None:
    event_handler, use_case, events = handler
    image = tmp_path / "gone.png"

    class Event:
        is_directory = False
        src_path = str(image)

    event_handler.on_deleted(Event())
    use_case.remove_indexed_file.assert_called_once()
    assert events[-1][1] == "removed"


def test_unchanged_file_emits_skipped(tmp_path: Path, handler) -> None:
    event_handler, use_case, events = handler
    use_case.index_changed_file.return_value = None
    image = tmp_path / "same.png"
    _write_png(image)

    class Event:
        is_directory = False
        src_path = str(image)

    with patch("src.core.use_cases.monitor_folder.validate_image", return_value=True):
        event_handler.on_created(Event())
        _wait_for_events(events)
    assert events[-1][1] == "skipped"


def test_inflight_filesystem_events_are_coalesced(tmp_path: Path, handler) -> None:
    """Second modify while first auto-index runs must not start a parallel pass."""
    import threading

    event_handler, use_case, events = handler
    image = tmp_path / "PGYS2319.jpg"
    _write_png(image)

    started = threading.Event()
    release = threading.Event()
    calls = {"n": 0}

    def slow_index(path):
        calls["n"] += 1
        started.set()
        release.wait(timeout=2.0)
        return 99

    use_case.index_changed_file.side_effect = slow_index

    class Event:
        is_directory = False
        src_path = str(image)

    with patch("src.core.use_cases.monitor_folder.validate_image", return_value=True):
        event_handler.on_modified(Event())
        assert started.wait(timeout=2.0)
        # Same file changes again while first pass is still settling/indexing.
        event_handler.on_modified(Event())
        time.sleep(0.2)
        assert calls["n"] == 1
        release.set()
        _wait_for_events(events, timeout=3.0)

    # Coalesced follow-up may run once more after the first finishes.
    assert calls["n"] in (1, 2)
    assert calls["n"] <= 2
    indexed = [e for e in events if e[1] == "indexed"]
    assert 1 <= len(indexed) <= 2
