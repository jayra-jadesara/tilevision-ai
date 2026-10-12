"""AutoIndexNotifier bridges watchdog threads to the Qt UI immediately."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from src.presentation.auto_index_notifier import AutoIndexNotifier


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


def test_notify_emits_indexed_immediately(qapp):
    notifier = AutoIndexNotifier()
    seen: list[tuple[str, str]] = []
    notifier.catalog_updated.connect(lambda path, action: seen.append((path, action)))

    notifier.notify("/tiles/a.jpg", "indexed", True)
    qapp.processEvents()

    assert seen == [("/tiles/a.jpg", "indexed")]


def test_notify_emits_failed_even_when_success_false(qapp):
    notifier = AutoIndexNotifier()
    seen: list[tuple[str, str]] = []
    notifier.catalog_updated.connect(lambda path, action: seen.append((path, action)))

    notifier.notify("/tiles/bad.jpg", "failed", False)
    qapp.processEvents()

    assert seen == [("/tiles/bad.jpg", "failed")]


def test_notify_skips_unchanged(qapp):
    notifier = AutoIndexNotifier()
    seen: list[tuple[str, str]] = []
    notifier.catalog_updated.connect(lambda path, action: seen.append((path, action)))

    notifier.notify("/tiles/same.jpg", "skipped", True)
    qapp.processEvents()

    assert seen == []


def test_notify_ignores_unsuccessful_index(qapp):
    notifier = AutoIndexNotifier()
    seen: list[tuple[str, str]] = []
    notifier.catalog_updated.connect(lambda path, action: seen.append((path, action)))

    notifier.notify("/tiles/x.jpg", "indexed", False)
    qapp.processEvents()

    assert seen == []
