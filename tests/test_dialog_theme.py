"""Tests for shared dialog theme QSS."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from src.presentation.views.update_dialog import UpdateAvailableDialog
from src.theme.theme_manager import get_dialog_qss
from src.utils.update_check import UpdateInfo


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_get_dialog_qss_includes_buttons_and_progress(theme):
    qss = get_dialog_qss(theme)
    assert "QDialog" in qss
    assert "#PrimaryButton" in qss
    assert "#DialogProgressBar" in qss
    assert "#LinkButton" in qss


def test_update_dialog_applies_theme_object_names(qapp):
    info = UpdateInfo(
        current_version="1.2.3",
        latest_version="1.2.6",
        release_notes="notes",
        download_url="https://example.com/setup.exe",
    )
    dialog = UpdateAvailableDialog(
        info,
        theme="light",
        auto_start_download=False,
        auto_install_after_download=False,
    )
    assert dialog._download_btn.objectName() == "PrimaryButton"
    assert dialog._later_btn.objectName() == "SecondaryButton"
    assert dialog._progress.objectName() == "DialogProgressBar"
    assert "PrimaryButton" in dialog.styleSheet()
    assert "DialogProgressBar" in dialog.styleSheet()
    dialog.close()
