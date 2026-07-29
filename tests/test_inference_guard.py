"""Tests for inference lock timeouts (search must not wait forever)."""

from __future__ import annotations

import threading
import time

import pytest

from src.ai.inference_guard import (
    InferenceBusyError,
    synchronized_inference,
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
