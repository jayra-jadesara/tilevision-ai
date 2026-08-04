"""Tests for themed message dialogs (Mac/Windows theme chrome)."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QPushButton

from src.presentation.dialogs import message_box
from src.presentation.dialogs.message_box import ThemedMessageDialog, resolve_theme
from src.theme.theme_manager import get_dialog_qss


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("kind", ["information", "warning", "critical", "question"])
def test_themed_message_dialog_applies_theme_qss(qapp, theme, kind):
    dialog = ThemedMessageDialog(
        None,
        "Test Title",
        "Body text for the themed dialog.",
        kind=kind,
        buttons=message_box.StandardButton.Ok,
        theme=theme,
    )
    qss = dialog.styleSheet()
    assert "PrimaryButton" in qss
    assert "ThemedMessageDialog" in qss or f"MessageIcon_{kind}" in qss
    assert get_dialog_qss(theme).split("{")[0][:20] in qss or "QDialog" in qss
    assert dialog.findChild(QPushButton, "PrimaryButton") is not None
    dialog.close()


def test_question_dialog_has_primary_and_secondary_buttons(qapp):
    dialog = ThemedMessageDialog(
        None,
        "Confirm",
        "Continue?",
        kind="question",
        buttons=message_box.StandardButton.Yes | message_box.StandardButton.No,
        default_button=message_box.StandardButton.No,
        theme="light",
    )
    yes = None
    no = None
    for btn in dialog.findChildren(QPushButton):
        if btn.text() == "Yes":
            yes = btn
        elif btn.text() == "No":
            no = btn
    assert yes is not None and yes.objectName() == "PrimaryButton"
    assert no is not None and no.objectName() == "SecondaryButton"
    assert no.isDefault()
    dialog.close()


def test_resolve_theme_from_parent_and_app_property(qapp):
    parent = QDialog()
    parent._theme = "light"
    assert resolve_theme(parent) == "light"

    orphan = QDialog()
    qapp.setProperty("tilevision_theme", "dark")
    assert resolve_theme(orphan) == "dark"
    qapp.setProperty("tilevision_theme", "light")
    parent.close()
    orphan.close()


def test_ok_click_returns_ok(qapp, monkeypatch):
    dialog = ThemedMessageDialog(
        None,
        "Done",
        "Saved.",
        kind="information",
        buttons=message_box.StandardButton.Ok,
        theme="dark",
    )
    # Avoid blocking: accept via button handler directly.
    dialog._on_button(message_box.StandardButton.Ok)
    assert dialog.result_button() == message_box.StandardButton.Ok
    dialog.close()
