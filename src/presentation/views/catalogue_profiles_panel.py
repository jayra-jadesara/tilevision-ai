"""Inline Export Catalogue company profile management (Settings tab)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QPushButton,
    QFormLayout,
    QFileDialog,
    QMessageBox,
    QSplitter,
)

from src.services.catalogue_master_service import CatalogueMaster, CatalogueMasterService
from src.theme.theme_manager import get_palette, get_shared_view_qss


class CatalogueProfilesPanel(QWidget):
    """List + editor for export PDF company profiles — no nested popups."""

    profiles_changed = Signal()

    def __init__(self, *, theme: str = "light", parent=None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._service = CatalogueMasterService()
        self._editing_id: Optional[str] = None
        self._build_ui()
        self._refresh_list()
        self._apply_styles()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)

        hint = QLabel(
            "Create company profiles here once, then pick them from the Export Catalogue "
            "dropdown on the Search page."
        )
        hint.setWordWrap(True)
        hint.setObjectName("PageSubtitle")
        layout.addWidget(hint)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Saved Profiles"))
        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_list_selection_changed)
        left_layout.addWidget(self._list)

        list_buttons = QHBoxLayout()
        self._new_btn = QPushButton("New")
        self._new_btn.setObjectName("SecondaryButton")
        self._new_btn.clicked.connect(self._on_new)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setObjectName("SecondaryButton")
        self._delete_btn.clicked.connect(self._on_delete)
        list_buttons.addWidget(self._new_btn)
        list_buttons.addWidget(self._delete_btn)
        list_buttons.addStretch()
        left_layout.addLayout(list_buttons)
        splitter.addWidget(left)

        right = QWidget()
        form_outer = QVBoxLayout(right)
        form_outer.setContentsMargins(0, 0, 0, 0)
        form_outer.addWidget(QLabel("Profile Details"))
        form = QFormLayout()

        self._display_name = QLineEdit()
        self._display_name.setPlaceholderText("e.g. ABC Ceramic")
        form.addRow("Profile Name", self._display_name)

        self._company_name = QLineEdit()
        form.addRow("Company Name", self._company_name)

        logo_row = QHBoxLayout()
        self._logo_path = QLineEdit()
        logo_browse = QPushButton("Browse")
        logo_browse.setObjectName("SecondaryButton")
        logo_browse.clicked.connect(self._browse_logo)
        logo_row.addWidget(self._logo_path)
        logo_row.addWidget(logo_browse)
        form.addRow("Company Logo", logo_row)

        self._email = QLineEdit()
        form.addRow("Email", self._email)
        self._phone = QLineEdit()
        form.addRow("Phone", self._phone)
        self._website = QLineEdit()
        form.addRow("Website", self._website)
        self._address = QLineEdit()
        form.addRow("Address", self._address)

        pdf_row = QHBoxLayout()
        self._pdf_folder = QLineEdit()
        self._pdf_folder.setPlaceholderText("Default folder for exported PDFs")
        pdf_browse = QPushButton("Browse")
        pdf_browse.setObjectName("SecondaryButton")
        pdf_browse.clicked.connect(self._browse_pdf_folder)
        pdf_row.addWidget(self._pdf_folder)
        pdf_row.addWidget(pdf_browse)
        form.addRow("Default PDF Folder", pdf_row)

        form_outer.addLayout(form)
        form_outer.addStretch()

        self._save_btn = QPushButton("Save Profile")
        self._save_btn.setObjectName("PrimaryButton")
        self._save_btn.clicked.connect(self._on_save)
        form_outer.addWidget(self._save_btn)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

    def _refresh_list(self, select_id: Optional[str] = None) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for master in sorted(self._service.masters, key=lambda m: m.display_name.lower()):
            item = QListWidgetItem(master.display_name)
            item.setData(Qt.ItemDataRole.UserRole, master.id)
            self._list.addItem(item)
            if select_id and master.id == select_id:
                self._list.setCurrentItem(item)
        self._list.blockSignals(False)

        if self._list.count() == 0:
            self._on_new()
        elif select_id is None and self._list.currentItem() is None:
            self._list.setCurrentRow(0)

    def _on_list_selection_changed(
        self,
        current: Optional[QListWidgetItem],
        _previous: Optional[QListWidgetItem],
    ) -> None:
        if current is None:
            return
        master_id = current.data(Qt.ItemDataRole.UserRole)
        master = self._service.get(str(master_id))
        if master is None:
            return
        self._editing_id = master.id
        self._load_form(master)

    def _load_form(self, master: CatalogueMaster) -> None:
        self._display_name.setText(master.display_name)
        self._company_name.setText(master.company_name)
        self._logo_path.setText(master.logo_path)
        self._email.setText(master.email)
        self._phone.setText(master.phone)
        self._website.setText(master.website)
        self._address.setText(master.address)
        self._pdf_folder.setText(master.default_pdf_folder)

    def _clear_form(self) -> None:
        self._editing_id = None
        for field in (
            self._display_name,
            self._company_name,
            self._logo_path,
            self._email,
            self._phone,
            self._website,
            self._address,
            self._pdf_folder,
        ):
            field.clear()
        self._list.clearSelection()

    def _form_master(self) -> CatalogueMaster:
        display_name = self._display_name.text().strip()
        company_name = self._company_name.text().strip() or display_name
        common = dict(
            display_name=display_name,
            company_name=company_name,
            logo_path=self._logo_path.text().strip(),
            email=self._email.text().strip(),
            phone=self._phone.text().strip(),
            website=self._website.text().strip(),
            address=self._address.text().strip(),
            default_pdf_folder=self._pdf_folder.text().strip(),
        )
        if self._editing_id:
            return CatalogueMaster(id=self._editing_id, **common)
        return CatalogueMaster(**common)

    def _on_new(self) -> None:
        self._clear_form()
        self._display_name.setFocus()

    def _on_save(self) -> None:
        try:
            master = self._form_master()
            if not master.display_name:
                raise ValueError("Profile name is required.")

            if self._editing_id and self._service.get(self._editing_id):
                saved = self._service.update(master)
            else:
                saved = self._service.add(master)
                self._editing_id = saved.id
        except ValueError as exc:
            QMessageBox.warning(self, "Save Failed", str(exc))
            return
        except KeyError:
            saved = self._service.add(self._form_master())
            self._editing_id = saved.id

        self._refresh_list(select_id=self._editing_id)
        self.profiles_changed.emit()

    def _on_delete(self) -> None:
        if not self._editing_id:
            QMessageBox.information(self, "Select Profile", "Select a profile to delete.")
            return
        master = self._service.get(self._editing_id)
        if master is None:
            return
        reply = QMessageBox.question(
            self,
            "Delete Profile",
            f"Delete profile \"{master.display_name}\"?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._service.delete(self._editing_id)
        self._editing_id = None
        self._refresh_list()
        self.profiles_changed.emit()

    def _browse_logo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Company Logo",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if path:
            self._logo_path.setText(path)

    def _browse_pdf_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Default PDF Folder")
        if folder:
            self._pdf_folder.setText(folder)

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        self._apply_styles()

    def _apply_styles(self) -> None:
        p = get_palette(self._theme)
        self.setStyleSheet(
            get_shared_view_qss(self._theme)
            + f"""
            QListWidget {{
                background-color: {p['bg_panel']};
                border: 1px solid {p['border']};
                border-radius: 8px;
                color: {p['text_primary']};
            }}
            QLineEdit {{
                background-color: {p['bg_input']};
                border: 1px solid {p['border_strong']};
                border-radius: 6px;
                padding: 6px 8px;
                color: {p['text_primary']};
            }}
            QLabel {{ color: {p['text_secondary']}; }}
            """
        )
