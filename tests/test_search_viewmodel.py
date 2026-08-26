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


def _drain_worker(vm, timeout=3.0) -> None:
    """Wait for SearchWorker QThread to finish so Windows teardown does not abort."""
    def _idle() -> bool:
        worker = getattr(vm, "_worker", None)
        return worker is None or not worker.isRunning()

    _pump_until(_idle, timeout=timeout)
    # Let deleteLater / queued slots settle before the test returns.
    for _ in range(10):
        QCoreApplication.processEvents()
        time.sleep(0.01)


@pytest.fixture(autouse=True)
def _qt_thread_settle(qapp):
    yield
    for _ in range(15):
        QCoreApplication.processEvents()
        time.sleep(0.01)

def _write_query(path: Path, color=(120, 80, 40)) -> Path:
    """Write a tiny valid JPEG so ViewModel preflight validation passes."""
    from PIL import Image

    Image.new("RGB", (16, 16), color=color).save(path, format="JPEG")
    return path


class FakeSearchUseCase:
    """Fake SearchTilesUseCase — returns canned results or raises, on demand."""

    def __init__(self, results=None, error=None, delay=0.0):
        self._results = results if results is not None else []
        self._error = error
        self._delay = delay
        self.calls = []

    def execute(self, query_image_path, top_k=20, filters=None, on_stage=None, query_origin=None):
        self.calls.append((query_image_path, top_k, filters))
        if self._delay:
            time.sleep(self._delay)
        if self._error:
            raise self._error
        return self._results

    def get_index_health(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            is_compatible=True,
            stale_count=0,
            indexed_count=getattr(self, "indexed_count", 1),
        )

    def get_searchable_count(self):
        return int(getattr(self, "searchable_count", getattr(self, "indexed_count", 1)))

    def get_filter_options(self):
        return getattr(self, "filter_options", {"brand": ["Acme"], "category": ["Floor"]})


def _make_result(score=90.0, path="/tmp/tile.jpg"):
    tile = TileImage(file_path=path, file_name="tile.jpg", file_size=1, dimensions="1x1")
    return SearchResult(tile=tile, similarity_score=score, thumbnail_path=path)


def test_successful_search_transitions_to_results(qapp, tmp_path):
    query_file = tmp_path / "query.jpg"
    _write_query(query_file)

    use_case = FakeSearchUseCase(results=[_make_result()])
    vm = SearchViewModel(use_case=use_case, default_top_k=20)

    states = []
    vm.state_changed.connect(states.append)

    vm.search_by_image(str(query_file))
    assert vm.state == SearchState.SEARCHING

    assert _pump_until(lambda: vm.state == SearchState.RESULTS)
    _drain_worker(vm)
    assert states == [SearchState.SEARCHING, SearchState.RESULTS]
    assert len(vm.last_results) == 1
    assert use_case.calls == [(str(query_file), 20, {})]


def test_empty_results_transitions_to_no_results_state(qapp, tmp_path):
    query_file = tmp_path / "query.jpg"
    _write_query(query_file)

    use_case = FakeSearchUseCase(results=[])
    vm = SearchViewModel(use_case=use_case)

    vm.search_by_image(str(query_file))
    assert _pump_until(lambda: vm.state == SearchState.NO_RESULTS)
    _drain_worker(vm)


def test_use_case_exception_transitions_to_error_state(qapp, tmp_path):
    query_file = tmp_path / "query.jpg"
    _write_query(query_file)

    use_case = FakeSearchUseCase(error=RuntimeError("model not loaded"))
    vm = SearchViewModel(use_case=use_case)

    errors = []
    vm.search_error.connect(errors.append)

    vm.search_by_image(str(query_file))
    assert _pump_until(lambda: vm.state == SearchState.ERROR)
    _drain_worker(vm)
    assert errors and "model not loaded" in errors[0]


def test_missing_query_file_does_not_start_a_worker(qapp, tmp_path):
    use_case = FakeSearchUseCase(results=[_make_result()])
    vm = SearchViewModel(use_case=use_case)

    vm.search_by_image(str(tmp_path / "does_not_exist.jpg"))

    assert vm.state == SearchState.ERROR
    assert use_case.calls == []  # never reached the use case


def test_corrupt_query_image_fails_fast_without_searching(qapp, tmp_path):
    corrupt = tmp_path / "not_an_image.jpg"
    corrupt.write_bytes(b"this is not an image payload")

    use_case = FakeSearchUseCase(results=[_make_result()])
    vm = SearchViewModel(use_case=use_case)
    errors = []
    vm.search_error.connect(errors.append)

    vm.search_by_image(str(corrupt))

    assert vm.state == SearchState.ERROR
    assert use_case.calls == []
    assert errors and "not a valid, readable image" in errors[0]
    assert "not_an_image.jpg" in errors[0]


def test_corrupt_query_while_searching_is_not_queued(qapp, tmp_path):
    query_file = tmp_path / "query.jpg"
    _write_query(query_file)
    corrupt = tmp_path / "not_an_image.jpg"
    corrupt.write_bytes(b"this is not an image payload")

    use_case = FakeSearchUseCase(results=[_make_result()], delay=0.25)
    vm = SearchViewModel(use_case=use_case)
    errors = []
    vm.search_error.connect(errors.append)

    vm.search_by_image(str(query_file))
    assert vm.state == SearchState.SEARCHING

    vm.search_by_image(str(corrupt))
    assert vm.state == SearchState.SEARCHING  # in-flight search continues
    assert errors and "not a valid, readable image" in errors[0]

    assert _pump_until(lambda: vm.state == SearchState.RESULTS, timeout=3.0)
    _drain_worker(vm)
    assert len(use_case.calls) == 1
    assert use_case.calls[0][0] == str(query_file)


def test_concurrent_search_request_is_queued_while_searching(qapp, tmp_path):
    query_file = tmp_path / "query.jpg"
    _write_query(query_file)
    second = tmp_path / "query2.jpg"
    _write_query(second, color=(40, 80, 120))

    # Slow enough that the second call definitely arrives while still searching.
    use_case = FakeSearchUseCase(results=[_make_result()], delay=0.25)
    vm = SearchViewModel(use_case=use_case)

    vm.search_by_image(str(query_file))
    assert vm.state == SearchState.SEARCHING

    vm.search_by_image(str(second))  # queued, not dropped
    assert _pump_until(lambda: len(use_case.calls) >= 2, timeout=5.0)
    assert _pump_until(lambda: vm.state == SearchState.RESULTS, timeout=3.0)
    _drain_worker(vm)
    assert use_case.calls[0][0] == str(query_file)
    assert use_case.calls[1][0] == str(second)

def test_clear_results_resets_to_idle(qapp, tmp_path):
    query_file = tmp_path / "query.jpg"
    _write_query(query_file)

    use_case = FakeSearchUseCase(results=[_make_result()])
    vm = SearchViewModel(use_case=use_case)

    vm.search_by_image(str(query_file))
    assert _pump_until(lambda: vm.state == SearchState.RESULTS)
    _drain_worker(vm)

    vm.clear_results()
    assert vm.state == SearchState.IDLE
    assert vm.last_results == []


# ── Task C: Search History integration ──────────────────────────────────


class FakeHistoryRepo:
    def __init__(self):
        self.recorded = []

    def record_search(self, query_image_path, result_count, elapsed_seconds=None, query_thumbnail_path=None):
        self.recorded.append((query_image_path, result_count, elapsed_seconds))

    def get_recent_searches(self, limit=10):
        return list(reversed(self.recorded))[:limit]

    def get_last_search(self):
        return self.recorded[-1] if self.recorded else None


class FakeActivityRepo:
    def __init__(self):
        self.recorded = []

    def record_activity(self, activity_type, message):
        self.recorded.append((activity_type, message))

    def get_recent_activity(self, limit=10):
        return list(reversed(self.recorded))[:limit]


def test_successful_search_is_recorded_in_history(qapp, tmp_path):
    query_file = tmp_path / "query.jpg"
    _write_query(query_file)

    history_repo = FakeHistoryRepo()
    use_case = FakeSearchUseCase(results=[_make_result(), _make_result()])
    vm = SearchViewModel(use_case=use_case, search_history_repository=history_repo)

    vm.search_by_image(str(query_file))
    assert _pump_until(lambda: vm.state == SearchState.RESULTS)
    _drain_worker(vm)

    assert len(history_repo.recorded) == 1
    assert history_repo.recorded[0][0] == str(query_file)
    assert history_repo.recorded[0][1] == 2


def test_successful_search_is_recorded_in_activity_log(qapp, tmp_path):
    query_file = tmp_path / "query.jpg"
    _write_query(query_file)

    activity_repo = FakeActivityRepo()
    use_case = FakeSearchUseCase(results=[_make_result()])
    vm = SearchViewModel(use_case=use_case, activity_log_repository=activity_repo)

    vm.search_by_image(str(query_file))
    assert _pump_until(lambda: vm.state == SearchState.RESULTS)
    _drain_worker(vm)

    assert len(activity_repo.recorded) == 1
    assert activity_repo.recorded[0][0] == "search"
    assert "query.jpg" in activity_repo.recorded[0][1]


def test_search_stats_signal_carries_count_and_elapsed_time(qapp, tmp_path):
    query_file = tmp_path / "query.jpg"
    _write_query(query_file)

    use_case = FakeSearchUseCase(results=[_make_result(), _make_result(), _make_result()])
    vm = SearchViewModel(use_case=use_case)

    captured = []
    vm.search_stats_ready.connect(lambda count, elapsed: captured.append((count, elapsed)))

    vm.search_by_image(str(query_file))
    assert _pump_until(lambda: vm.state == SearchState.RESULTS)
    _drain_worker(vm)

    assert len(captured) == 1
    assert captured[0][0] == 3
    assert captured[0][1] >= 0


def test_repeat_search_reruns_with_existing_file(qapp, tmp_path):
    query_file = tmp_path / "query.jpg"
    _write_query(query_file)

    use_case = FakeSearchUseCase(results=[_make_result()])
    vm = SearchViewModel(use_case=use_case)

    vm.repeat_search(str(query_file))
    assert _pump_until(lambda: vm.state == SearchState.RESULTS)
    _drain_worker(vm)
    assert use_case.calls == [(str(query_file), 20, {})]


def test_repeat_search_with_missing_file_emits_error(qapp, tmp_path):
    use_case = FakeSearchUseCase(results=[_make_result()])
    vm = SearchViewModel(use_case=use_case)

    errors = []
    vm.search_error.connect(errors.append)

    vm.repeat_search(str(tmp_path / "gone.jpg"))
    assert errors
    assert use_case.calls == []


def test_empty_index_fails_fast_without_worker(qapp, tmp_path):
    query_file = tmp_path / "query.jpg"
    _write_query(query_file)

    use_case = FakeSearchUseCase(results=[_make_result()])
    use_case.indexed_count = 0
    vm = SearchViewModel(use_case=use_case)

    errors = []
    statuses = []
    vm.search_error.connect(errors.append)
    vm.status_message.connect(statuses.append)

    vm.search_by_image(str(query_file))

    assert vm.state == SearchState.NO_RESULTS
    assert use_case.calls == []
    assert errors and "indexed" in errors[0].lower()
    assert any("indexed" in s.lower() for s in statuses)


def test_empty_faiss_with_indexed_tiles_fails_fast(qapp, tmp_path):
    query_file = tmp_path / "query.jpg"
    _write_query(query_file)

    use_case = FakeSearchUseCase(results=[_make_result()])
    use_case.indexed_count = 300
    use_case.searchable_count = 0
    vm = SearchViewModel(use_case=use_case)

    errors = []
    vm.search_error.connect(errors.append)

    vm.search_by_image(str(query_file))

    assert vm.state == SearchState.ERROR
    assert use_case.calls == []
    assert errors and "rebuild" in errors[0].lower()


def test_search_notifies_busy_callback(qapp, tmp_path):
    query_file = tmp_path / "query.jpg"
    _write_query(query_file)

    busy_events = []
    use_case = FakeSearchUseCase(results=[_make_result()])
    vm = SearchViewModel(
        use_case=use_case,
        on_search_busy_changed=busy_events.append,
    )

    vm.search_by_image(str(query_file))
    assert busy_events == [True]
    assert _pump_until(lambda: vm.state == SearchState.RESULTS)
    _drain_worker(vm)
    assert busy_events == [True, False]


def test_search_timeout_stops_endless_searching(qapp, tmp_path):
    # Do not start a real QThread — destroying a still-running worker aborts.
    # Explicit timeout enables the optional abort path (disabled in production).
    use_case = FakeSearchUseCase(results=[_make_result()])
    vm = SearchViewModel(use_case=use_case, search_timeout_ms=45_000)

    errors = []
    vm.search_error.connect(errors.append)

    vm._last_query_path = str(tmp_path / "query.jpg")
    vm._set_state(SearchState.SEARCHING)
    vm._on_search_timeout()
    assert vm.state == SearchState.ERROR
    assert errors and "too long" in errors[0].lower()


def test_production_default_does_not_auto_abort_search(qapp):
    use_case = FakeSearchUseCase(results=[_make_result()])
    vm = SearchViewModel(use_case=use_case)
    assert vm._search_timeout_ms == 0
    errors = []
    vm.search_error.connect(errors.append)
    vm._set_state(SearchState.SEARCHING)
    vm._on_search_timeout()  # no-op when timeout disabled
    assert vm.state == SearchState.SEARCHING
    assert errors == []


def test_progress_resets_hang_watchdog(qapp):
    use_case = FakeSearchUseCase(results=[_make_result()])
    vm = SearchViewModel(use_case=use_case)
    assert hasattr(vm, "_stall_timer")
    vm._set_state(SearchState.SEARCHING)
    vm._stall_timer.start(60_000)
    vm._on_search_progress("Running AI match…")
    # Progress must restart the stall timer (still active / rearmed).
    assert vm._stall_timer.isActive()
    assert vm._stall_timer.remainingTime() > 0


def test_heartbeat_resets_stall_watchdog(qapp):
    use_case = FakeSearchUseCase(results=[_make_result()])
    vm = SearchViewModel(use_case=use_case)
    vm._set_state(SearchState.SEARCHING)
    vm._stall_timer.start(60_000)
    vm._on_search_heartbeat()
    assert vm._stall_timer.isActive()
    assert vm._stall_timer.remainingTime() > 0


def test_hang_abort_only_when_unresponsive(qapp, tmp_path):
    use_case = FakeSearchUseCase(results=[_make_result()])
    vm = SearchViewModel(use_case=use_case)
    errors = []
    vm.search_error.connect(errors.append)
    vm._last_query_path = str(tmp_path / "query.jpg")
    vm._set_state(SearchState.SEARCHING)
    vm._on_search_stall()
    assert vm.state == SearchState.ERROR
    assert errors and "stopped responding" in errors[0].lower()


def test_set_filter_noop_does_not_research(qapp, tmp_path):
    """Clearing a filter that was not active must not start a search."""
    query_file = tmp_path / "query.jpg"
    _write_query(query_file)
    use_case = FakeSearchUseCase(results=[_make_result()])
    vm = SearchViewModel(use_case=use_case)
    vm.search_by_image(str(query_file))
    assert _pump_until(lambda: vm.state == SearchState.RESULTS)
    _drain_worker(vm)
    assert len(use_case.calls) == 1

    empty_emits = []
    vm.results_ready.connect(lambda r: empty_emits.append(list(r)))
    vm.set_filter("brand", "")
    QCoreApplication.processEvents()
    assert len(use_case.calls) == 1
    assert vm.state == SearchState.RESULTS
    assert all(len(r) > 0 for r in empty_emits) or empty_emits == []


def test_catalog_filter_refresh_does_not_clear_displayed_results(qapp, tmp_path):
    """
    SISCON repro: auto-index → load_filter_options must not wipe results.

    Simulates MainWindow._on_catalog_changed → load_filter_options after a
    successful search, including a stale brand value that disappears.
    """
    query_file = tmp_path / "query.jpg"
    _write_query(query_file)
    use_case = FakeSearchUseCase(results=[_make_result(path=str(tmp_path / "PGYS2319.jpg"))])
    use_case.filter_options = {
        "brand": ["Acme"],
        "category": ["Floor"],
        "color": [],
        "size": [],
    }
    vm = SearchViewModel(use_case=use_case)
    vm.search_by_image(str(query_file))
    assert _pump_until(lambda: vm.state == SearchState.RESULTS)
    _drain_worker(vm)
    assert len(vm.last_results) == 1

    # User had a brand filter that the catalog refresh will no longer list.
    vm._active_filters["brand"] = "GoneBrand"
    empty_payloads = []
    vm.results_ready.connect(lambda r: empty_payloads.append(list(r)))

    # Quiet drop (what SearchView._on_filters_available does now).
    vm.drop_filter_quietly("brand")
    vm.load_filter_options()
    QCoreApplication.processEvents()

    assert vm.state == SearchState.RESULTS
    assert len(vm.last_results) == 1
    assert "brand" not in vm.active_filters
    assert len(use_case.calls) == 1  # no auto re-search
    assert not any(len(p) == 0 for p in empty_payloads)


def test_health_check_indexed_zero_logs_reason(qapp, tmp_path):
    query_file = tmp_path / "query.jpg"
    _write_query(query_file)
    use_case = FakeSearchUseCase(results=[_make_result()])
    use_case.indexed_count = 0
    vm = SearchViewModel(use_case=use_case)
    empties = []
    vm.results_ready.connect(lambda r: empties.append(list(r)))
    vm.search_by_image(str(query_file))
    QCoreApplication.processEvents()
    assert empties and empties[0] == []
    assert vm.state == SearchState.NO_RESULTS
    assert use_case.calls == []  # never started worker
