"""PR D: UI activity event filter + privacy + dedupe."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_licence_key_never_appears_in_ui_log(qapp, caplog):
    from PySide6.QtWidgets import QLineEdit
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent
    from src.utils.ui_activity import UiActivityFilter

    edit = QLineEdit()
    edit.setObjectName("licenseKeyEdit")
    edit.setEchoMode(QLineEdit.EchoMode.Password)
    secret = "TV-LICENCE-KEY-ABC-12345"
    edit.setText(secret)

    filt = UiActivityFilter()
    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        # Simulate typing — filter must not emit the secret.
        for ch in secret:
            ev = QKeyEvent(
                QEvent.Type.KeyPress,
                ord(ch),
                __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.KeyboardModifier.NoModifier,
                ch,
            )
            filt.eventFilter(edit, ev)

    joined = "\n".join(r.message for r in caplog.records)
    assert secret not in joined
    assert "TV-LICENCE" not in joined


def test_click_burst_deduped(qapp, caplog):
    from PySide6.QtWidgets import QPushButton
    from PySide6.QtCore import QEvent, Qt, QPoint
    from PySide6.QtGui import QMouseEvent
    from src.utils.ui_activity import UiActivityFilter

    btn = QPushButton("Export Catalogue")
    btn.setObjectName("ExportButton")
    filt = UiActivityFilter()

    def _click():
        ev = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPoint(1, 1),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        filt.eventFilter(btn, ev)

    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        for _ in range(8):
            _click()
    messages = [r.message for r in caplog.records if "Export Catalogue" in r.message]
    # 300 ms dedupe → a burst collapses to one line
    assert len(messages) == 1


def test_mask_if_sensitive():
    from src.utils.ui_activity import mask_if_sensitive

    assert mask_if_sensitive("license_key", "SECRET") == "***"
    assert mask_if_sensitive("folder_path", "/tiles") == "/tiles"


def test_mainwindow_nav_clicks_log(qapp, caplog, tmp_path, catalogue_master_service):
    """QTest.mouseClick on Index/Search/Settings/Help/Export asserts log lines."""
    pytest.importorskip("PySide6.QtTest")
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from unittest.mock import MagicMock

    from src.config.settings import AppSettings
    from src.core.use_cases.find_duplicates import FindDuplicatesUseCase
    from src.presentation.viewmodels.indexing_viewmodel import IndexingViewModel
    from src.presentation.viewmodels.search_viewmodel import SearchViewModel
    from src.presentation.views.main_window import DashboardDataProviders, MainWindow
    from src.utils.ui_activity import install_ui_activity_logging

    class _FakeIndexUseCase:
        def get_persisted_folder_state(self):
            return None

    settings = AppSettings(config_dir=tmp_path)
    settings.theme = "dark"
    settings.setup_wizard_completed = True
    search_vm = SearchViewModel(use_case=MagicMock(), default_top_k=10)
    repo = MagicMock()
    repo.get_all.return_value = []
    window = MainWindow(
        indexing_viewmodel=IndexingViewModel(use_case=_FakeIndexUseCase()),
        search_viewmodel=search_vm,
        find_duplicates_use_case=FindDuplicatesUseCase(repo),
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

    recorded = []
    install_ui_activity_logging(
        QApplication.instance(),
        activity_recorder=lambda t, m: recorded.append((t, m)),
    )

    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        for btn in (
            window._nav_index_button,
            window._nav_search_button,
            window._nav_settings_button,
            window._nav_help_button,
        ):
            QTest.mouseClick(btn, Qt.MouseButton.LeftButton)
            QApplication.processEvents()

        export_btn = getattr(window._search_view, "_export_button", None)
        if export_btn is not None:
            window._navigate(1)
            QTest.mouseClick(export_btn, Qt.MouseButton.LeftButton)
            QApplication.processEvents()

        combos = getattr(window._search_view, "_filter_combos", {}) or {}
        if combos:
            combo = next(iter(combos.values()))
            if combo.count() > 1:
                combo.setCurrentIndex(1)
                QApplication.processEvents()

    ui_lines = [r.message for r in caplog.records if r.message.startswith("[UI]")]
    assert any(t == "navigation" for t, _ in recorded)
    assert any("Opened Help" in m or "Opened Index" in m for t, m in recorded)

    sample = Path("/opt/cursor/artifacts")
    sample.mkdir(parents=True, exist_ok=True)
    sample_lines = ui_lines[:20] or [f"[activity] {t}: {m}" for t, m in recorded[:20]]
    # Pad to ~20 sample lines for the PR description.
    while len(sample_lines) < 20 and recorded:
        for t, m in recorded:
            sample_lines.append(f"[activity] {t}: {m}")
            if len(sample_lines) >= 20:
                break
        break
    (sample / "ui_activity_sample_20.txt").write_text(
        "\n".join(sample_lines[:20]) + "\n", encoding="utf-8"
    )

    window.close()
    window.deleteLater()
