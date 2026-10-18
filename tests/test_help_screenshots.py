"""Help page screenshots + drift guard for button labels named in Help text."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.presentation.views.help_view import (
    HelpView,
    _STEPS,
    help_button_labels_named_in_text,
)

_HELP_DIR = Path(__file__).resolve().parents[1] / "src" / "resources" / "help"


def _expected_stems() -> list[str]:
    return [step.screenshot_stem for step in _STEPS]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_help_step_screenshots_exist_wide(theme: str):
    for stem in _expected_stems():
        path = _HELP_DIR / theme / f"{stem}.png"
        assert path.is_file(), f"Missing help screenshot: {path}"
        assert path.stat().st_size > 10_000, f"Screenshot too small: {path}"
        with Image.open(path) as img:
            assert img.size[0] >= 1400, f"{path} width {img.size[0]} < 1400"
            assert img.size[1] >= 300


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_help_view_renders(theme: str, qapp):
    view = HelpView(theme=theme)
    view.resize(1440, 900)
    view.show()
    qapp.processEvents()
    assert view.isVisible()
    view.close()


def test_help_button_label_drift_guard(qapp, tmp_path, catalogue_master_service):
    """Fail if Help names a button label that no longer exists in the UI."""
    from unittest.mock import MagicMock

    from PySide6.QtWidgets import QApplication, QAbstractButton, QWidget

    from src.config.settings import AppSettings
    from src.core.use_cases.find_duplicates import FindDuplicatesUseCase
    from src.presentation.viewmodels.indexing_viewmodel import IndexingViewModel
    from src.presentation.viewmodels.search_viewmodel import SearchViewModel
    from src.presentation.views.main_window import DashboardDataProviders, MainWindow

    class _FakeIndexUseCase:
        def get_persisted_folder_state(self):
            return None

    settings = AppSettings(config_dir=tmp_path)
    settings.theme = "light"
    settings.setup_wizard_completed = True
    window = MainWindow(
        indexing_viewmodel=IndexingViewModel(use_case=_FakeIndexUseCase()),
        search_viewmodel=SearchViewModel(use_case=MagicMock(), default_top_k=10),
        find_duplicates_use_case=FindDuplicatesUseCase(MagicMock()),
        settings=settings,
        catalog_count_provider=lambda: 0,
        catalogue_master_service=catalogue_master_service,
        dashboard_providers=DashboardDataProviders(
            indexed_folder_count=lambda: 0,
            database_size=lambda: 0,
            faiss_size=lambda: 0,
            last_search=lambda: None,
            recent_activity=lambda: [],
            recent_searches=lambda: [],
        ),
        license_details={
            "license_type": "1-Year",
            "customer_name": "Test",
            "is_trial": False,
        },
        on_check_updates=lambda: None,
    )
    window.show()
    QApplication.processEvents()

    texts: set[str] = set()

    def _collect(w: QWidget) -> None:
        if isinstance(w, QAbstractButton):
            t = (w.text() or "").replace("&", "").strip()
            if t:
                texts.add(t)
        for child in w.findChildren(QWidget):
            if isinstance(child, QAbstractButton):
                t = (child.text() or "").replace("&", "").strip()
                if t:
                    texts.add(t)

    _collect(window)
    # Also gather tooltips that act as icon-only actions.
    for child in window.findChildren(QWidget):
        tip = (child.toolTip() or "").strip()
        if tip:
            texts.add(tip)

    missing = []
    for label in help_button_labels_named_in_text():
        if not any(label.casefold() in t.casefold() for t in texts):
            missing.append(label)
    assert not missing, f"Help names UI labels not found: {missing}"

    window.close()
    window.deleteLater()
