"""
Tests for Settings page fixes from user feedback:
  - Search Results Shown is now a preset dropdown (5/10/15/20/25), not a
    free-form spinbox.
  - Thumbnail Size control removed from the UI (now automatic/hidden).
  - Database Path / Thumbnail Cache path no longer shown in Overview.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _install_fake_ai_deps():
    if "torch" not in sys.modules:
        fake_torch = types.ModuleType("torch")

        class _FakeCuda:
            @staticmethod
            def is_available():
                return False

        fake_torch.cuda = _FakeCuda()
        sys.modules["torch"] = fake_torch
    if "open_clip" not in sys.modules:
        sys.modules["open_clip"] = types.ModuleType("open_clip")


_install_fake_ai_deps()

PySide6 = pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from src.config.settings import AppSettings
from src.presentation.views.settings_view import SettingsView


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_search_results_dropdown_has_preset_values(qapp, tmp_path):
    settings = AppSettings(config_dir=tmp_path)
    sv = SettingsView(settings=settings)

    options = [sv._top_k_combo.itemText(i) for i in range(sv._top_k_combo.count())]
    assert options == ["5", "10", "15", "20", "25"]


def test_selecting_a_preset_updates_settings(qapp, tmp_path):
    settings = AppSettings(config_dir=tmp_path)
    sv = SettingsView(settings=settings)

    sv._top_k_combo.setCurrentText("15")
    assert settings.top_k == 15


def test_non_preset_existing_value_is_preserved_not_silently_changed(qapp, tmp_path):
    settings = AppSettings(config_dir=tmp_path)
    settings.top_k = 42  # not one of the presets

    sv = SettingsView(settings=settings)
    assert sv._top_k_combo.currentText() == "42"
    assert settings.top_k == 42  # unchanged just from opening Settings


def test_thumbnail_size_control_no_longer_exists(qapp, tmp_path):
    settings = AppSettings(config_dir=tmp_path)
    sv = SettingsView(settings=settings)
    assert not hasattr(sv, "_thumbnail_size_spinbox")


def test_overview_no_longer_displays_raw_file_paths(qapp, tmp_path):
    settings = AppSettings(config_dir=tmp_path)
    sv = SettingsView(settings=settings, db_path_provider=lambda: Path("/some/db/path.db"))

    # Overview labels must not leak raw thumbnail cache dir or database paths.
    from PySide6.QtWidgets import QGroupBox, QLabel

    overview_boxes = [
        box for box in sv.findChildren(QGroupBox) if box.title() == "Overview"
    ]
    assert len(overview_boxes) == 1
    label_texts = [lbl.text() for lbl in overview_boxes[0].findChildren(QLabel)]
    assert not any("thumbnails" in t.lower() for t in label_texts)
    assert not any("path.db" in t for t in label_texts)


def test_backup_still_works_without_path_being_displayed(qapp, tmp_path):
    """db_path_provider is still used internally for Backup Database, even
    though it's no longer shown as a label."""
    settings = AppSettings(config_dir=tmp_path)
    sv = SettingsView(settings=settings, db_path_provider=lambda: Path("/some/db/path.db"))
    assert sv._backup_button.isEnabled()
