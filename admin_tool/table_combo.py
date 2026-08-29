"""Combo box that works reliably inside QTableWidget cells (Windows/Qt)."""

from __future__ import annotations

from typing import Callable, Iterable

from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QComboBox, QTableWidget


class TableComboBox(QComboBox):
    """Dropdown that opens on click inside table cells."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.InsertAtBottom)
        self.setMaxVisibleItems(14)
        self.setMinimumContentsLength(12)
        self.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self.showPopup()
        super().mousePressEvent(event)


_NONE_LABEL = "(none)"


def populate_combo(
    combo: TableComboBox,
    options: Iterable[str],
    current: str,
    *,
    allow_none: bool = False,
) -> None:
    combo.blockSignals(True)
    combo.clear()
    seen: set[str] = set()
    if allow_none:
        combo.addItem(_NONE_LABEL)
        seen.add(_NONE_LABEL)
    for value in options:
        text = str(value).strip()
        if not text and allow_none:
            continue
        if text in seen:
            continue
        seen.add(text)
        combo.addItem(text)
    display = current.strip() if current else (_NONE_LABEL if allow_none else "")
    if display and display not in seen:
        combo.addItem(display)
    combo.setCurrentText(display if display else (_NONE_LABEL if allow_none else ""))
    combo.blockSignals(False)


def combo_value(combo: TableComboBox, *, allow_none: bool = False) -> str:
    text = combo.currentText().strip()
    if allow_none and text == _NONE_LABEL:
        return ""
    return text


def attach_table_combo(
    table: QTableWidget,
    row: int,
    column: int,
    options: list[str],
    current: str,
    *,
    allow_none: bool = False,
    on_remember: Callable[[str], None] | None = None,
) -> TableComboBox:
    combo = TableComboBox(table)
    populate_combo(combo, options, current, allow_none=allow_none)

    def _remember(text: str) -> None:
        if on_remember is None:
            return
        value = text.strip()
        if allow_none and value == _NONE_LABEL:
            value = ""
        if value:
            on_remember(value)

    combo.currentTextChanged.connect(_remember)
    table.setCellWidget(row, column, combo)
    return combo
