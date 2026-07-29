"""Tests for inference lock timeouts and search priority yield."""

from __future__ import annotations

import threading
import time

import pytest

from src.ai.inference_guard import (
    InferenceBusyError,
    begin_search_priority,
    end_search_priority,
    search_priority_active,
    synchronized_inference,
    wait_while_search_priority,
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
