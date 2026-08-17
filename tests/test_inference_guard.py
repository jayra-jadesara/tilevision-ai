"""Tests for inference lock timeouts and search priority yield."""

from __future__ import annotations

import threading
import time

import pytest

from src.ai.inference_guard import (
    InferenceBusyError,
    begin_search_priority,
    end_search_priority,
    inference_lock_held,
    interactive_cpu_thread_count,
    is_warmup_compute,
    search_priority_active,
    synchronized_inference,
    wait_while_search_priority,
    warmup_compute_scope,
    warmup_in_progress,
)


def test_timed_acquire_raises_when_lock_held():
    held = threading.Event()
    release = threading.Event()

    def holder():
        with synchronized_inference(timeout=5.0, purpose="holder"):
            held.set()
            release.wait(timeout=5.0)

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    assert held.wait(timeout=2.0)

    with pytest.raises(InferenceBusyError, match="busy"):
        with synchronized_inference(timeout=0.2, purpose="search"):
            pass

    release.set()
    thread.join(timeout=2.0)


def test_timed_acquire_succeeds_when_free():
    with synchronized_inference(timeout=1.0, purpose="search"):
        time.sleep(0.01)


def test_search_priority_blocks_indexing_until_cleared():
    begin_search_priority()
    try:
        assert search_priority_active()
        started = time.monotonic()
        # Short wait: should return after max_wait, not hang forever.
        wait_while_search_priority(max_wait_s=0.15)
        assert time.monotonic() - started >= 0.1
    finally:
        end_search_priority()
    assert not search_priority_active()
    wait_while_search_priority(max_wait_s=0.5)


def test_wait_until_inference_idle_returns_when_free():
    from src.ai.inference_guard import wait_until_inference_idle

    assert wait_until_inference_idle(max_wait_s=0.5) is True


def test_wait_until_inference_idle_times_out_while_held():
    from src.ai.inference_guard import synchronized_inference, wait_until_inference_idle

    held = threading.Event()
    release = threading.Event()

    def holder():
        with synchronized_inference(timeout=5.0, purpose="holder"):
            held.set()
            release.wait(timeout=5.0)

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    assert held.wait(timeout=2.0)
    assert wait_until_inference_idle(max_wait_s=0.25) is False
    release.set()
    thread.join(timeout=2.0)


def test_indexing_yields_while_search_active_then_continues():
    begin_search_priority()
    resumed = threading.Event()

    def indexer():
        wait_while_search_priority(max_wait_s=5.0)
        resumed.set()

    thread = threading.Thread(target=indexer, daemon=True)
    thread.start()
    time.sleep(0.05)
    assert not resumed.is_set()
    end_search_priority()
    assert resumed.wait(timeout=2.0)
    thread.join(timeout=2.0)


def test_dinov2_lock_blocks_faiss_ntotal_like_windows_repro():
    """The 44s Search-priority-ON → Starting-worker gap: FAISS ntotal waits on DINOv2 lock."""
    held = threading.Event()
    release = threading.Event()

    def holder():
        with synchronized_inference(timeout=5.0, purpose="DINOv2 embed"):
            held.set()
            release.wait(timeout=5.0)

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    assert held.wait(timeout=2.0)
    assert inference_lock_held()

    started = time.monotonic()
    with pytest.raises(InferenceBusyError):
        with synchronized_inference(timeout=0.35, purpose="FAISS ntotal"):
            pass
    assert time.monotonic() - started >= 0.3

    release.set()
    thread.join(timeout=2.0)


def test_warmup_compute_scope_does_not_hold_inference_lock():
    """Warmup mid-forward must not block FAISS ntotal / catalog-cache-hit search."""
    in_scope = threading.Event()
    release = threading.Event()

    def warmup():
        with warmup_compute_scope(torch_threads=1):
            assert is_warmup_compute()
            assert warmup_in_progress()
            in_scope.set()
            release.wait(timeout=5.0)

    thread = threading.Thread(target=warmup, daemon=True)
    thread.start()
    assert in_scope.wait(timeout=2.0)
    assert warmup_in_progress()
    assert not inference_lock_held()

    started = time.monotonic()
    with synchronized_inference(timeout=1.0, purpose="FAISS ntotal"):
        waited = time.monotonic() - started
    assert waited < 0.25

    release.set()
    thread.join(timeout=2.0)
    assert not warmup_in_progress()


def test_warmup_caps_torch_threads_while_search_is_active():
    torch = pytest.importorskip("torch")

    previous = int(torch.get_num_threads())
    torch.set_num_threads(max(2, previous))
    try:
        with warmup_compute_scope(torch_threads=1):
            assert torch.get_num_threads() == 1
            begin_search_priority()
            try:
                assert torch.get_num_threads() == 1
            finally:
                pass
        assert torch.get_num_threads() == interactive_cpu_thread_count()
    finally:
        end_search_priority()
        torch.set_num_threads(previous)


def test_search_priority_restores_torch_threads_when_warmup_idle():
    torch = pytest.importorskip("torch")

    previous = int(torch.get_num_threads())
    torch.set_num_threads(1)
    try:
        begin_search_priority()
        try:
            assert torch.get_num_threads() == interactive_cpu_thread_count()
        finally:
            end_search_priority()
    finally:
        torch.set_num_threads(previous)
