"""Prove macOS Intel search uses OpenMP-safe Python AI threads."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.presentation.workers import native_ai_thread as nat
from src.presentation.workers.search_worker import SearchWorker


@pytest.fixture(autouse=True)
def _reset_native_ai_install(monkeypatch):
    monkeypatch.setattr(nat, "_INSTALLED", False)
    # Clear class patch flags between tests (Search + Indexing only).
    from src.presentation.workers.indexing_worker import IndexingWorker

    for cls in (SearchWorker, IndexingWorker):
        if hasattr(cls, "_tv_python_ai_thread_patched"):
            delattr(cls, "_tv_python_ai_thread_patched")
    yield
    monkeypatch.setattr(nat, "_INSTALLED", False)


def test_should_use_python_ai_threads_on_darwin(monkeypatch):
    monkeypatch.setattr(nat.sys, "platform", "darwin")
    monkeypatch.delenv("TILEVISION_FORCE_QTHREAD_AI", raising=False)
    monkeypatch.delenv("TILEVISION_FORCE_PYTHON_AI_THREADS", raising=False)
    assert nat.should_use_python_ai_threads() is True


def test_should_keep_qthread_on_windows(monkeypatch):
    monkeypatch.setattr(nat.sys, "platform", "win32")
    monkeypatch.delenv("TILEVISION_FORCE_PYTHON_AI_THREADS", raising=False)
    assert nat.should_use_python_ai_threads() is False


def test_force_qthread_override(monkeypatch):
    monkeypatch.setattr(nat.sys, "platform", "darwin")
    monkeypatch.setenv("TILEVISION_FORCE_QTHREAD_AI", "1")
    assert nat.should_use_python_ai_threads() is False


def test_configure_macos_openmp_caps_intel(monkeypatch):
    monkeypatch.setattr(nat.sys, "platform", "darwin")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    with patch("src.utils.platform_info.is_mac_intel", return_value=True):
        for key in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            monkeypatch.delenv(key, raising=False)
        nat.configure_macos_openmp_for_ai()
    assert nat.os.environ.get("OMP_NUM_THREADS") == "1"
    assert nat.os.environ.get("MKL_NUM_THREADS") == "1"


def test_install_patches_search_worker_start_on_darwin(monkeypatch):
    monkeypatch.setattr(nat.sys, "platform", "darwin")
    monkeypatch.delenv("TILEVISION_FORCE_QTHREAD_AI", raising=False)
    nat.install_python_ai_worker_threads()
    assert getattr(SearchWorker, "_tv_python_ai_thread_patched", False) is True
    assert nat.production_uses_python_ai_threads() is True
