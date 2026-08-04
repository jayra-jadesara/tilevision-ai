"""
Themed modal message dialogs — light/dark TileVision chrome on Mac & Windows.

Native QMessageBox uses OS chrome that ignores app QSS (especially on macOS).
These helpers mirror the QMessageBox static API but render via QDialog +
get_dialog_qss, matching UpdateAvailableDialog and the rest of the app.
"""

from __future__ import annotations

from typing import Optional, Union

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtWidgets import QMessageBox as _QtMessageBox

from src.theme.theme_manager import get_dialog_qss, get_palette

# Re-export so call sites can use message_box.StandardButton like QMessageBox.
StandardButton = _QtMessageBox.StandardButton
Icon = _QtMessageBox.Icon

_BUTTON_LABELS = {
    StandardButton.Ok: "OK",
    StandardButton.Yes: "Yes",
    StandardButton.No: "No",
    StandardButton.Cancel: "Cancel",
    StandardButton.Close: "Close",
    StandardButton.Abort: "Abort",
    StandardButton.Retry: "Retry",
    StandardButton.Ignore: "Ignore",
    StandardButton.Save: "Save",
    StandardButton.Discard: "Discard",
    StandardButton.Apply: "Apply",
    StandardButton.Reset: "Reset",
    StandardButton.RestoreDefaults: "Restore Defaults",
    StandardButton.Help: "Help",
    StandardButton.Open: "Open",
    StandardButton.SaveAll: "Save All",
    StandardButton.YesToAll: "Yes to All",
    StandardButton.NoToAll: "No to All",
}

_PRIMARY_BUTTONS = frozenset(
    {
        StandardButton.Ok,
        StandardButton.Yes,
        StandardButton.Save,
        StandardButton.Apply,
        StandardButton.Retry,
        StandardButton.Open,
        StandardButton.YesToAll,
    }
)

_KIND_META = {
    "information": ("ℹ", "Info"),
    "warning": ("!", "Warning"),
    "critical": ("✕", "Error"),
    "question": ("?", "Question"),
}


def resolve_theme(parent: Optional[QWidget] = None) -> str:
    """Resolve light/dark from parent chain, then QApplication property."""
    widget: Optional[QWidget] = parent
    while widget is not None:
        for attr in ("_theme", "_current_theme"):
            value = getattr(widget, attr, None)
            if value in ("light", "dark"):
                return value
        widget = widget.parentWidget() if hasattr(widget, "parentWidget") else None

    app = QApplication.instance()
    if app is not None:
        prop = app.property("tilevision_theme")
        if prop in ("light", "dark"):
            return prop
    return "dark"


def _iter_standard_buttons(buttons: StandardButton):
    """Yield individual StandardButton flags set on a button combination."""
    # Prefer a stable, customer-friendly order.
    preferred = (
        StandardButton.YesToAll,
        StandardButton.Yes,
        StandardButton.Ok,
        StandardButton.Save,
        StandardButton.SaveAll,
        StandardButton.Open,
        StandardButton.Apply,
        StandardButton.Retry,
        StandardButton.Ignore,
        StandardButton.Reset,
        StandardButton.RestoreDefaults,
        StandardButton.Help,
        StandardButton.NoToAll,
        StandardButton.No,
        StandardButton.Discard,
        StandardButton.Abort,
        StandardButton.Cancel,
        StandardButton.Close,
    )
    found = []
    for button in preferred:
        if buttons & button:
            found.append(button)
    # Any remaining flags not in the preferred list.
    remaining = int(buttons)
    for button in found:
        remaining &= ~int(button)
    if remaining:
        # Fallback: scan known labels.
        for button in _BUTTON_LABELS:
            if remaining & int(button) and button not in found:
                found.append(button)
                remaining &= ~int(button)
    return found


class ThemedMessageDialog(QDialog):
    """Modal message dialog styled with the active TileVision theme."""

    def __init__(
        self,
        parent: Optional[QWidget],
        title: str,
        text: str,
        *,
        kind: str = "information",
        buttons: StandardButton = StandardButton.Ok,
        default_button: StandardButton = StandardButton.NoButton,
        theme: Optional[str] = None,
    ) -> None:
        super().__init__(parent)
        self._result_button = StandardButton.Cancel
        self._theme = theme if theme in ("light", "dark") else resolve_theme(parent)
        self.setObjectName("ThemedMessageDialog")
        self.setModal(True)
        self.setWindowTitle(title or _KIND_META.get(kind, ("", "Message"))[1])
        self.setMinimumWidth(420)
        self.setMaximumWidth(560)

        glyph, _fallback_title = _KIND_META.get(kind, ("ℹ", "Message"))
        palette = get_palette(self._theme)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 20)
        root.setSpacing(16)

        header = QHBoxLayout()
        header.setSpacing(14)

        icon = QLabel(glyph)
        icon.setObjectName(f"MessageIcon_{kind}")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setFixedSize(40, 40)
        header.addWidget(icon, alignment=Qt.AlignmentFlag.AlignTop)

        text_col = QVBoxLayout()
        text_col.setSpacing(8)
        title_label = QLabel(title)
        title_label.setObjectName("DialogTitle")
        title_label.setWordWrap(True)
        text_col.addWidget(title_label)

        body = QLabel(text)
        body.setObjectName("MessageBody")
        body.setWordWrap(True)
        body.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        text_col.addWidget(body)
        header.addLayout(text_col, stretch=1)
        root.addLayout(header)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.setSpacing(10)

        button_flags = list(_iter_standard_buttons(buttons))
        if not button_flags:
            button_flags = [StandardButton.Ok]

        if default_button == StandardButton.NoButton:
            # Prefer a primary affirmative as default.
            for candidate in (
                StandardButton.Ok,
                StandardButton.Yes,
                StandardButton.Save,
                StandardButton.Retry,
            ):
                if candidate in button_flags:
                    default_button = candidate
                    break
            if default_button == StandardButton.NoButton:
                default_button = button_flags[0]

        self._buttons: dict[StandardButton, QPushButton] = {}
        for flag in button_flags:
            btn = QPushButton(_BUTTON_LABELS.get(flag, str(flag)))
            is_primary = flag in _PRIMARY_BUTTONS
            btn.setObjectName("PrimaryButton" if is_primary else "SecondaryButton")
            btn.setDefault(flag == default_button)
            btn.setAutoDefault(flag == default_button)
            btn.clicked.connect(lambda _checked=False, f=flag: self._on_button(f))
            button_row.addWidget(btn)
            self._buttons[flag] = btn

        root.addLayout(button_row)

        # Kind-colored icon chips on top of shared dialog QSS.
        kind_colors = {
            "information": (palette["accent"], palette["highlight_bg"], palette["accent_text"]),
            "warning": (palette["warning_text"], palette["warning_bg"], palette["warning_text"]),
            "critical": (palette["danger_text"], palette["danger_bg"], palette["danger_text"]),
            "question": (palette["accent"], palette["highlight_bg"], palette["accent_text"]),
        }
        border, bg, fg = kind_colors.get(
            kind, (palette["accent"], palette["highlight_bg"], palette["accent_text"])
        )
        extra = f"""
        #ThemedMessageDialog #MessageIcon_information,
        #ThemedMessageDialog #MessageIcon_warning,
        #ThemedMessageDialog #MessageIcon_critical,
        #ThemedMessageDialog #MessageIcon_question {{
            background-color: {bg};
            color: {fg};
            border: 1px solid {border};
            border-radius: 20px;
            font-size: 18px;
            font-weight: 700;
        }}
        #ThemedMessageDialog #MessageBody {{
            color: {palette['text_secondary']};
            font-size: 13px;
            line-height: 1.35;
        }}
        """
        self.setStyleSheet(get_dialog_qss(self._theme) + extra)

        default_btn = self._buttons.get(default_button)
        if default_btn is not None:
            default_btn.setFocus(Qt.FocusReason.OtherFocusReason)

    def _on_button(self, flag: StandardButton) -> None:
        self._result_button = flag
        if flag in (
            StandardButton.Yes,
            StandardButton.Ok,
            StandardButton.Save,
            StandardButton.Apply,
            StandardButton.Retry,
            StandardButton.Open,
            StandardButton.YesToAll,
        ):
            self.accept()
        else:
            self.reject()

    def result_button(self) -> StandardButton:
        return self._result_button


def _show(
    kind: str,
    parent: Optional[QWidget],
    title: str,
    text: str,
    buttons: StandardButton = StandardButton.Ok,
    default_button: StandardButton = StandardButton.NoButton,
) -> StandardButton:
    dialog = ThemedMessageDialog(
        parent,
        title,
        text,
        kind=kind,
        buttons=buttons,
        default_button=default_button,
    )
    dialog.exec()
    return dialog.result_button()


def information(
    parent: Optional[QWidget],
    title: str,
    text: str,
    buttons: StandardButton = StandardButton.Ok,
    defaultButton: StandardButton = StandardButton.NoButton,
) -> StandardButton:
    return _show("information", parent, title, text, buttons, defaultButton)


def warning(
    parent: Optional[QWidget],
    title: str,
    text: str,
    buttons: StandardButton = StandardButton.Ok,
    defaultButton: StandardButton = StandardButton.NoButton,
) -> StandardButton:
    return _show("warning", parent, title, text, buttons, defaultButton)


def critical(
    parent: Optional[QWidget],
    title: str,
    text: str,
    buttons: StandardButton = StandardButton.Ok,
    defaultButton: StandardButton = StandardButton.NoButton,
) -> StandardButton:
    return _show("critical", parent, title, text, buttons, defaultButton)


def question(
    parent: Optional[QWidget],
    title: str,
    text: str,
    buttons: Union[StandardButton, int] = StandardButton.Yes | StandardButton.No,
    defaultButton: StandardButton = StandardButton.NoButton,
) -> StandardButton:
    return _show("question", parent, title, text, StandardButton(buttons), defaultButton)
