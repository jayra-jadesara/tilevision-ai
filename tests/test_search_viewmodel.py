"""
Tests for SearchViewModel / SearchWorker (Feature 2: AI Tile Search).

Uses a fake SearchTilesUseCase (no torch/open_clip/faiss needed) driven
through a real PySide6 QApplication event loop via QSignalSpy-style waiting,
so the QThread worker's signals genuinely cross threads as they would in
production.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PySide6 = pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from src.core.models import SearchResult, TileImage
from src.presentation.viewmodels.search_viewmodel import SearchViewModel, SearchState


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _pump_until(condition, timeout=5.0):
    """Process Qt events until condition() is True or timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


class FakeSearchUseCase:
    """Fake SearchTilesUseCase — returns canned results or raises, on demand."""

    def __init__(self, results=None, error=None, delay=0.0):
        self._results = results if results is not None else []
        self._error = error
        self._delay = delay
        self.calls = []

    def execute(self, query_image_path, top_k=20):
        self.calls.append((query_image_path, top_k))
        if self._delay:
            time.sleep(self._delay)
        if self._error:
            raise self._error
        return self._results


def _make_result(score=90.0, path="/tmp/tile.jpg"):
    tile = TileImage(file_path=path, file_name="tile.jpg", file_size=1, dimensions="1x1")
    return SearchResult(tile=tile, similarity_score=score, thumbnail_path=path)


def test_successful_search_transitions_to_results(qapp, tmp_path):
    query_file = tmp_path / "query.jpg"
    query_file.write_bytes(b"fake")

    use_case = FakeSearchUseCase(results=[_make_result()])
    vm = SearchViewModel(use_case=use_case, default_top_k=20)

    states = []
    vm.state_changed.connect(states.append)

    vm.search_by_image(str(query_file))
    assert vm.state == SearchState.SEARCHING

    assert _pump_until(lambda: vm.state == SearchState.RESULTS)
    assert states == [SearchState.SEARCHING, SearchState.RESULTS]
    assert len(vm.last_results) == 1
    assert use_case.calls == [(str(query_file), 20)]


def test_empty_results_transitions_to_no_results_state(qapp, tmp_path):
    query_file = tmp_path / "query.jpg"
    query_file.write_bytes(b"fake")

    use_case = FakeSearchUseCase(results=[])
    vm = SearchViewModel(use_case=use_case)

    vm.search_by_image(str(query_file))
    assert _pump_until(lambda: vm.state == SearchState.NO_RESULTS)


def test_use_case_exception_transitions_to_error_state(qapp, tmp_path):
    query_file = tmp_path / "query.jpg"
    query_file.write_bytes(b"fake")

    use_case = FakeSearchUseCase(error=RuntimeError("model not loaded"))
    vm = SearchViewModel(use_case=use_case)

    errors = []
    vm.search_error.connect(errors.append)

    vm.search_by_image(str(query_file))
    assert _pump_until(lambda: vm.state == SearchState.ERROR)
    assert errors and "model not loaded" in errors[0]


def test_missing_query_file_does_not_start_a_worker(qapp, tmp_path):
    use_case = FakeSearchUseCase(results=[_make_result()])
    vm = SearchViewModel(use_case=use_case)

    vm.search_by_image(str(tmp_path / "does_not_exist.jpg"))

    assert vm.state == SearchState.ERROR
    assert use_case.calls == []  # never reached the use case


def test_concurrent_search_request_is_ignored_while_searching(qapp, tmp_path):
    query_file = tmp_path / "query.jpg"
    query_file.write_bytes(b"fake")

    # Slow enough that the second call definitely arrives while still searching.
    use_case = FakeSearchUseCase(results=[_make_result()], delay=0.3)
    vm = SearchViewModel(use_case=use_case)

    vm.search_by_image(str(query_file))
    assert vm.state == SearchState.SEARCHING

    vm.search_by_image(str(query_file))  # should be ignored
    assert _pump_until(lambda: vm.state == SearchState.RESULTS, timeout=3.0)

    # Only one call ever reached the use case, despite two search_by_image calls.
    assert len(use_case.calls) == 1


def test_clear_results_resets_to_idle(qapp, tmp_path):
    query_file = tmp_path / "query.jpg"
    query_file.write_bytes(b"fake")

    use_case = FakeSearchUseCase(results=[_make_result()])
    vm = SearchViewModel(use_case=use_case)

    vm.search_by_image(str(query_file))
    assert _pump_until(lambda: vm.state == SearchState.RESULTS)

    vm.clear_results()
    assert vm.state == SearchState.IDLE
    assert vm.last_results == []
