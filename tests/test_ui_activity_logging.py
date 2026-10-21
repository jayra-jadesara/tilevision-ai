"""PR D: UI activity event filter + privacy + dedupe + signal hooks."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
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


@pytest.fixture
def ui_filter(qapp):
    """Fresh filter (not the process-wide install) for unit tests."""
    from src.utils.ui_activity import UiActivityFilter

    return UiActivityFilter()


def _ui_messages(caplog) -> list[str]:
    return [r.message for r in caplog.records if r.message.startswith("[UI]")]


def test_licence_key_never_appears_in_ui_log(qapp, ui_filter, caplog):
    from PySide6.QtWidgets import QLineEdit
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    edit = QLineEdit()
    edit.setObjectName("licenseKeyEdit")
    edit.setEchoMode(QLineEdit.EchoMode.Password)
    secret = "TV-LICENCE-KEY-ABC-12345"
    edit.setText(secret)

    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        for ch in secret:
            ev = QKeyEvent(
                QEvent.Type.KeyPress,
                ord(ch),
                Qt.KeyboardModifier.NoModifier,
                ch,
            )
            ui_filter.eventFilter(edit, ev)

    joined = "\n".join(r.message for r in caplog.records)
    assert secret not in joined
    assert "TV-LICENCE" not in joined


def test_click_burst_deduped(qapp, ui_filter, caplog):
    from PySide6.QtWidgets import QPushButton
    from PySide6.QtCore import QEvent, Qt, QPoint
    from PySide6.QtGui import QMouseEvent

    btn = QPushButton("Export Catalogue")
    btn.setObjectName("ExportButton")

    def _click():
        ev = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPoint(1, 1),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        ui_filter.eventFilter(btn, ev)

    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        for _ in range(8):
            _click()
    messages = [m for m in _ui_messages(caplog) if "Export Catalogue" in m]
    assert len(messages) == 1


def test_mask_if_sensitive():
    from src.utils.ui_activity import mask_if_sensitive

    assert mask_if_sensitive("license_key", "SECRET") == "***"
    assert mask_if_sensitive("folder_path", "/tiles") == "/tiles"


def test_button_click_logs(qapp, ui_filter, caplog):
    from PySide6.QtWidgets import QPushButton
    from PySide6.QtCore import QEvent, Qt, QPoint
    from PySide6.QtGui import QMouseEvent

    btn = QPushButton("Start Indexing")
    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        ev = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPoint(1, 1),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        ui_filter.eventFilter(btn, ev)
    assert any("Start Indexing" in m for m in _ui_messages(caplog))


def test_action_triggered_logs(qapp, ui_filter, caplog):
    """QAction.triggered signal hook (not nonexistent QEvent.Type.Triggered)."""
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QWidget
    from PySide6.QtCore import QEvent

    host = QWidget()
    action = QAction("Reveal in Finder", host)
    host.addAction(action)
    ui_filter._maybe_hook_signals(host)

    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        action.trigger()
    assert any("Reveal in Finder" in m and "QAction" in m for m in _ui_messages(caplog))


def test_tab_change_logs(qapp, ui_filter, caplog):
    from PySide6.QtWidgets import QTabWidget, QWidget

    tabs = QTabWidget()
    tabs.addTab(QWidget(), "General")
    tabs.addTab(QWidget(), "Export Profiles")
    ui_filter._maybe_hook_signals(tabs)
    # Also walk children the way Show/ChildAdded would — must not double-hook
    # the internal QTabBar.
    for child in tabs.findChildren(QWidget):
        ui_filter._maybe_hook_signals(child)

    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        tabs.setCurrentIndex(1)
    tab_msgs = [m for m in _ui_messages(caplog) if "tab '" in m and "Export Profiles" in m]
    assert len(tab_msgs) == 1, tab_msgs
    assert "QTabWidget" in tab_msgs[0]


def test_bare_modifier_not_logged(qapp, ui_filter, caplog):
    """Ctrl alone must not emit 'Ctrl+Control' (or any shortcut line)."""
    from PySide6.QtWidgets import QWidget
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    w = QWidget()
    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        for key in (
            Qt.Key.Key_Control,
            Qt.Key.Key_Alt,
            Qt.Key.Key_Meta,
            Qt.Key.Key_Shift,
        ):
            mods = Qt.KeyboardModifier.NoModifier
            if key == Qt.Key.Key_Control:
                mods = Qt.KeyboardModifier.ControlModifier
            elif key == Qt.Key.Key_Alt:
                mods = Qt.KeyboardModifier.AltModifier
            elif key == Qt.Key.Key_Meta:
                mods = Qt.KeyboardModifier.MetaModifier
            elif key == Qt.Key.Key_Shift:
                mods = Qt.KeyboardModifier.ShiftModifier
            ev = QKeyEvent(QEvent.Type.KeyPress, key, mods, "")
            ui_filter.eventFilter(w, ev)
    assert not any("shortcut" in m for m in _ui_messages(caplog))


def test_custom_context_menu_logs_action_once(qapp, ui_filter, caplog):
    """CustomContextMenu + QAction → one action line (no context-menu / menu open)."""
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QMenu, QWidget
    from PySide6.QtCore import QEvent, Qt

    w = QWidget()
    w.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    menu = QMenu(w)  # untitled → transient
    action = QAction("Copy Path", menu)
    menu.addAction(action)
    ui_filter._maybe_hook_signals(menu)

    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        ui_filter.eventFilter(w, QEvent(QEvent.Type.ContextMenu))
        ui_filter.eventFilter(menu, QEvent(QEvent.Type.Show))
        action.trigger()
    msgs = _ui_messages(caplog)
    assert len([m for m in msgs if "Copy Path" in m]) == 1
    assert not any("context-menu" in m for m in msgs)
    assert not any("menu open" in m for m in msgs)


def test_combo_activated_logs(qapp, ui_filter, caplog):
    from PySide6.QtWidgets import QComboBox

    combo = QComboBox()
    combo.setObjectName("BrandFilter")
    combo.addItems(["All", "Acme", "Beta"])
    ui_filter._maybe_hook_signals(combo)

    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        combo.activated.emit(1)
    assert any("Acme" in m and "combo" in m for m in _ui_messages(caplog))


def test_dialog_open_close_logs(qapp, ui_filter, caplog):
    from PySide6.QtWidgets import QDialog
    from PySide6.QtCore import QEvent

    dlg = QDialog()
    dlg.setWindowTitle("Crop Tool")
    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        ui_filter.eventFilter(dlg, QEvent(QEvent.Type.Show))
        ui_filter.eventFilter(dlg, QEvent(QEvent.Type.Close))
    msgs = _ui_messages(caplog)
    assert any("dialog open" in m and "Crop Tool" in m for m in msgs)
    assert any("dialog close" in m and "Crop Tool" in m for m in msgs)


def test_context_menu_logs(qapp, ui_filter, caplog):
    """Default context-menu policy still logs the ContextMenu event once."""
    from PySide6.QtWidgets import QWidget
    from PySide6.QtCore import QEvent, Qt

    w = QWidget()
    w.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        ui_filter.eventFilter(w, QEvent(QEvent.Type.ContextMenu))
    assert len([m for m in _ui_messages(caplog) if "context-menu" in m]) == 1


def test_shortcut_logs(qapp, ui_filter, caplog):
    from PySide6.QtWidgets import QWidget
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    w = QWidget()
    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        ev = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_F,
            Qt.KeyboardModifier.ControlModifier,
            "f",
        )
        ui_filter.eventFilter(w, ev)
    assert any("shortcut" in m for m in _ui_messages(caplog))


def test_drop_logs(qapp, ui_filter, caplog):
    from PySide6.QtWidgets import QWidget
    from PySide6.QtCore import QEvent

    w = QWidget()
    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        ui_filter.eventFilter(w, QEvent(QEvent.Type.Drop))
    assert any("drop onto" in m for m in _ui_messages(caplog))


def test_stack_change_logs_without_activity_nav(qapp, ui_filter, caplog):
    from PySide6.QtWidgets import QStackedWidget, QWidget

    recorded: list[tuple[str, str]] = []
    ui_filter.set_activity_recorder(lambda t, m: recorded.append((t, m)))
    stack = QStackedWidget()
    a = QWidget()
    a.setObjectName("IndexPage")
    b = QWidget()
    b.setObjectName("SearchPage")
    stack.addWidget(a)
    stack.addWidget(b)
    ui_filter._maybe_hook_signals(stack)

    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        stack.setCurrentIndex(1)
    assert any("SearchPage" in m for m in _ui_messages(caplog))
    # Filter must not emit Activity "Opened …" — MainWindow owns that.
    assert not any(t == "navigation" and m.startswith("Opened ") for t, m in recorded)


def test_nonexistent_qevent_types_do_not_exist():
    """Guard: PySide6 must not gain these (or we can simplify the filter)."""
    from PySide6.QtCore import QEvent

    assert not hasattr(QEvent.Type, "Triggered")
    assert not hasattr(QEvent.Type, "CurrentChanged")


def test_mainwindow_nav_clicks_log(qapp, caplog, tmp_path, catalogue_master_service):
    """QTest.mouseClick on Index/Search/Settings/Help asserts log + single Activity nav."""
    pytest.importorskip("PySide6.QtTest")
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

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

    recorded: list[tuple[str, str]] = []
    from src.presentation.views.main_window import HistoryActivityProviders

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
        history_activity_providers=HistoryActivityProviders(
            activity_recorder=lambda t, m: recorded.append((t, m)),
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

    install_ui_activity_logging(
        QApplication.instance(),
        activity_recorder=lambda t, m: recorded.append((t, m)),
    )

    recorded.clear()
    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        for btn in (
            window._nav_index_button,
            window._nav_search_button,
            window._nav_settings_button,
            window._nav_help_button,
        ):
            QTest.mouseClick(btn, Qt.MouseButton.LeftButton)
            QApplication.processEvents()

    ui_lines = _ui_messages(caplog)
    assert any("click" in m and ("Index" in m or "Search" in m or "Help" in m) for m in ui_lines)

    nav_opened = [m for t, m in recorded if t == "navigation" and m.startswith("Opened ")]
    # Exactly one Activity line per nav click — filter must not double with
    # MainWindow._navigate (was: filter "Opened Index" AND _navigate).
    assert nav_opened.count("Opened Index") == 1
    assert nav_opened.count("Opened Search") == 1
    assert nav_opened.count("Opened Settings") == 1
    assert nav_opened.count("Opened Help") == 1

    export_btn = getattr(window._search_view, "_export_button", None)
    if export_btn is not None:
        with caplog.at_level(logging.INFO, logger="tilevision.ui"):
            window._navigate(1)
            QTest.mouseClick(export_btn, Qt.MouseButton.LeftButton)
            QApplication.processEvents()
        assert any("Export Catalogue" in m for m in _ui_messages(caplog))

    # Write sample under a writable path (never hardcode /opt/cursor — CI macOS
    # cannot create it → PermissionError). Prefer env override, else tmp.
    sample_root = Path(
        os.environ.get("TILEVISION_ARTIFACT_DIR")
        or os.environ.get("CURSOR_ARTIFACT_DIR")
        or tmp_path
    )
    sample_root.mkdir(parents=True, exist_ok=True)
    sample_lines = ui_lines[:20] or [f"[activity] {t}: {m}" for t, m in recorded[:20]]
    while len(sample_lines) < 20 and recorded:
        for t, m in recorded:
            sample_lines.append(f"[activity] {t}: {m}")
            if len(sample_lines) >= 20:
                break
        break
    (sample_root / "ui_activity_sample_20.txt").write_text(
        "\n".join(sample_lines[:20]) + "\n", encoding="utf-8"
    )

    window.close()
    window.deleteLater()


def _make_main_window(tmp_path, catalogue_master_service, recorded=None):
    from src.config.settings import AppSettings
    from src.core.use_cases.find_duplicates import FindDuplicatesUseCase
    from src.presentation.viewmodels.indexing_viewmodel import IndexingViewModel
    from src.presentation.viewmodels.search_viewmodel import SearchViewModel
    from src.presentation.views.main_window import (
        DashboardDataProviders,
        HistoryActivityProviders,
        MainWindow,
    )

    class _FakeIndexUseCase:
        def get_persisted_folder_state(self):
            return None

    settings = AppSettings(config_dir=tmp_path)
    settings.theme = "dark"
    settings.setup_wizard_completed = True
    repo = MagicMock()
    repo.get_all.return_value = []
    rec = recorded if recorded is not None else []
    return MainWindow(
        indexing_viewmodel=IndexingViewModel(use_case=_FakeIndexUseCase()),
        search_viewmodel=SearchViewModel(use_case=MagicMock(), default_top_k=10),
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
        history_activity_providers=HistoryActivityProviders(
            activity_recorder=lambda t, m: rec.append((t, m)),
        ),
        license_details={
            "license_type": "1-Year",
            "customer_name": "Test",
            "is_trial": False,
        },
        on_check_updates=lambda: None,
    )


def test_mainwindow_tab_context_shortcut_each_one_ui_line(
    qapp, caplog, tmp_path, catalogue_master_service
):
    """One Settings tab change, one context-menu action, one shortcut → one line each."""
    pytest.importorskip("PySide6.QtTest")
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QAction, QKeyEvent
    from PySide6.QtWidgets import QApplication, QMenu

    from src.utils.ui_activity import install_ui_activity_logging

    window = _make_main_window(tmp_path, catalogue_master_service)
    window.show()
    QApplication.processEvents()
    install_ui_activity_logging(QApplication.instance())

    # --- Settings tab change ---
    window._navigate(3)  # Settings
    QApplication.processEvents()
    tabs = window._settings_view._tabs
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        tabs.setCurrentIndex(1)
        QApplication.processEvents()
    tab_lines = [m for m in _ui_messages(caplog) if m.startswith("[UI] tab ")]
    assert len(tab_lines) == 1, tab_lines

    # --- Context menu action on Search results (CustomContextMenu path) ---
    window._navigate(1)
    QApplication.processEvents()
    table = window._search_view._results_table
    menu = QMenu(table)
    action = QAction("Copy Path", menu)
    menu.addAction(action)
    # Hook the way install_ui_activity_logging would when the menu shows.
    filt = install_ui_activity_logging(QApplication.instance())
    filt._maybe_hook_signals(action)  # direct QAction hook (macOS-safe)
    filt._maybe_hook_signals(menu)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        filt.eventFilter(table, QEvent(QEvent.Type.ContextMenu))
        filt.eventFilter(menu, QEvent(QEvent.Type.Show))
        action.trigger()
        QApplication.processEvents()
    action_lines = [m for m in _ui_messages(caplog) if "Copy Path" in m]
    assert len(action_lines) == 1, _ui_messages(caplog)
    assert not any("context-menu" in m for m in _ui_messages(caplog))
    assert not any("menu open" in m for m in _ui_messages(caplog))

    # --- One real shortcut ---
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        ev = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_F,
            Qt.KeyboardModifier.ControlModifier,
            "f",
        )
        filt.eventFilter(window, ev)
        QApplication.processEvents()
    shortcut_lines = [m for m in _ui_messages(caplog) if "shortcut" in m]
    assert len(shortcut_lines) == 1, shortcut_lines
    assert "Ctrl+Control" not in shortcut_lines[0]

    window.close()
    window.deleteLater()


def test_licence_plain_text_edit_masked_in_shortcut(qapp, ui_filter, caplog):
    """Licence QPlainTextEdit (real LicenseKeyEdit) must not leak key material."""
    from PySide6.QtWidgets import QPlainTextEdit
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    edit = QPlainTextEdit()
    edit.setObjectName("LicenseKeyEdit")
    secret = "TV-LICENCE-KEY-PLAIN-99999"
    edit.setPlainText(secret)

    with caplog.at_level(logging.INFO, logger="tilevision.ui"):
        ev = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_A,
            Qt.KeyboardModifier.ControlModifier,
            "a",
        )
        ui_filter.eventFilter(edit, ev)
    joined = "\n".join(r.message for r in caplog.records)
    assert secret not in joined
    assert "TV-LICENCE" not in joined
    assert not any("shortcut" in m for m in _ui_messages(caplog))
