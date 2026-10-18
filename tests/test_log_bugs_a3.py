"""
PR A3: Cancel-in-finished noise, reconcile log quieting, stall query label.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.use_cases.monitor_folder import TileImageEventHandler
from src.presentation.viewmodels.indexing_viewmodel import (
    IndexingState,
    IndexingViewModel,
)


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(40, 120, 80)).save(path, format="PNG")


def test_cancel_when_finished_does_not_warn(caplog):
    vm = IndexingViewModel(use_case=MagicMock())
    vm._set_state(IndexingState.FINISHED)
    with caplog.at_level(logging.WARNING, logger="tilevision.presentation.viewmodels.indexing_viewmodel"):
        vm.cancel_indexing()
    assert not any("Cancel called in invalid state" in r.message for r in caplog.records)


def test_cancel_when_idle_is_debug_only(caplog):
    vm = IndexingViewModel(use_case=MagicMock())
    assert vm.state == IndexingState.IDLE
    with caplog.at_level(logging.DEBUG, logger="tilevision.presentation.viewmodels.indexing_viewmodel"):
        vm.cancel_indexing()
    assert any("Cancel ignored" in r.message for r in caplog.records)
    assert not any(r.levelno >= logging.WARNING and "Cancel" in r.message for r in caplog.records)


def test_reconcile_unchanged_logs_debug_not_info(tmp_path, caplog):
    use_case = MagicMock()
    use_case.index_changed_file.return_value = None
    handler = TileImageEventHandler(indexing_use_case=use_case)
    image = tmp_path / "same.png"
    _write_png(image)

    with caplog.at_level(logging.DEBUG, logger="tilevision.core.use_cases.monitor_folder"):
        handler.reconcile_existing_file(str(image))

    info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert not any("Reconcile" in m and "same.png" in m for m in info_msgs)
    assert any("Reconcile skipped unchanged" in r.message for r in caplog.records)


def test_reconcile_indexed_logs_info(tmp_path, caplog):
    use_case = MagicMock()
    use_case.index_changed_file.return_value = 42
    handler = TileImageEventHandler(indexing_use_case=use_case)
    image = tmp_path / "new.png"
    _write_png(image)

    with caplog.at_level(logging.INFO, logger="tilevision.core.use_cases.monitor_folder"):
        handler.reconcile_existing_file(str(image))

    assert any("Reconcile indexed: new.png" in r.message for r in caplog.records)


def test_reconcile_discards_coalesced_watchdog_echo(tmp_path):
    """FS events during reconcile must not schedule a second pass."""
    use_case = MagicMock()
    use_case.index_changed_file.return_value = None
    handler = TileImageEventHandler(
        indexing_use_case=use_case,
        debounce_seconds=5.0,
    )
    image = tmp_path / "echo.png"
    _write_png(image)
    key = str(image.resolve())

    # Simulate a watchdog echo mid-reconcile by injecting into reprocess_after
    # while inflight (reconcile itself marks inflight).
    original = handler._use_case.index_changed_file

    def _index_and_echo(path):
        with handler._timer_lock:
            assert key in handler._inflight_paths
            handler._reprocess_after.add(key)
        return original(path)

    handler._use_case.index_changed_file = _index_and_echo
    handler.reconcile_existing_file(str(image))

    with handler._timer_lock:
        assert key not in handler._inflight_paths
        assert key not in handler._reprocess_after
        assert key not in handler._pending_timers


def test_stall_query_label_uses_description_text(qapp, caplog):
    from src.presentation.viewmodels.search_viewmodel import SearchViewModel, SearchState

    class FakeImageUC:
        def get_filter_options(self):
            return {}

        def get_index_health(self):
            return SimpleNamespace(is_compatible=True, stale_count=0, indexed_count=1)

        def get_searchable_count(self):
            return 1

    vm = SearchViewModel(use_case=FakeImageUC(), description_use_case=object())
    vm._active_search_kind = "description"
    vm._last_query_path = None
    vm._last_description_query = "matte gray subway tile"
    vm._set_state(SearchState.SEARCHING)

    with caplog.at_level(logging.ERROR, logger="tilevision.presentation.viewmodels.search_viewmodel"):
        vm._on_search_stall()

    assert any("matte gray subway tile" in r.message for r in caplog.records)
    assert not any("query=None" in r.message for r in caplog.records)


def test_clip_download_offline_message(monkeypatch, tmp_path):
    import huggingface_hub

    from src.ai.clip_embedder import ClipEmbedder

    out = tmp_path / "clip-vit-base-patch32"
    out.mkdir()
    monkeypatch.setattr(
        "src.ai.clip_model_paths.bundled_clip_model_dir",
        lambda: out,
    )

    def _fail(**kwargs):
        raise ConnectionError("Network unreachable")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fail)

    with pytest.raises(RuntimeError, match="internet connection") as exc_info:
        ClipEmbedder._download_to_bundled("openai/clip-vit-base-patch32")
    assert "description-search model" in str(exc_info.value).lower()
