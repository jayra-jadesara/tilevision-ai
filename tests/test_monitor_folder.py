"""Tests for auto folder monitoring event handling."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.use_cases.monitor_folder import TileImageEventHandler


def _write_png(path: Path) -> None:
    Image.new("RGB", (8, 8), color=(120, 80, 40)).save(path, format="PNG")


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
        import time

        time.sleep(0.35)

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
        import time

        time.sleep(0.35)
    assert events[-1][1] == "skipped"
