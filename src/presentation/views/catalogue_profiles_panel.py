"""Inline Export Catalogue company profile management (Settings tab)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal, QRegularExpression
from PySide6.QtGui import QRegularExpressionValidator
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
    QGroupBox,
    QCheckBox,
    QSpinBox,
    QFrame,
    QScrollArea,
)

from src.services.catalogue_master_service import CatalogueMaster, CatalogueMasterService
from src.theme.theme_manager import get_shared_view_qss, get_settings_view_qss
from src.utils.profile_validation import (
    validate_email,
    validate_logo_path,
    validate_phone,
    validate_profile_name,
    validate_website,
)


class CatalogueProfilesPanel(QWidget):
    """List + editor for export PDF company profiles — no nested popups."""

    profiles_changed = Signal()

    def __init__(self, *, theme: str = "light", parent=None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._service = CatalogueMasterService()
        self._editing_id: Optional[str] = None
        self._field_errors: dict[str, QLabel] = {}
        self._build_ui()
        self._refresh_list()
        self._apply_styles()

    def _build_ui(self) -> None:
        self.setObjectName("CatalogueProfilesPanel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        hint = QLabel(
            "Configure company details and export options once here. On Search, pick a profile "
            "and export — no extra editing needed."
        )
        hint.setWordWrap(True)
        hint.setObjectName("PageSubtitle")
        outer.addWidget(hint)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QFrame()
        left.setObjectName("ProfileSidebarCard")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)

        left_header = QHBoxLayout()
        left_header.addWidget(QLabel("Saved Profiles"))
        left_header.addStretch()
        self._new_btn = QPushButton("+ New")
        self._new_btn.setObjectName("ToolbarButton")
        self._new_btn.clicked.connect(self._on_new)
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setObjectName("ToolbarButton")
        self._delete_btn.clicked.connect(self._on_delete)
        left_header.addWidget(self._new_btn)
        left_header.addWidget(self._delete_btn)
        left_layout.addLayout(left_header)

        self._list = QListWidget()
        self._list.setObjectName("ProfileList")
        self._list.currentItemChanged.connect(self._on_list_selection_changed)
        left_layout.addWidget(self._list, stretch=1)
        splitter.addWidget(left)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setObjectName("ProfileEditorScroll")

        editor = QWidget()
        editor.setObjectName("ProfileEditorPage")
        editor_layout = QVBoxLayout(editor)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(12)

        company_box = QGroupBox("Company Details")
        company_box.setObjectName("SettingsSection")
        company_form = QFormLayout(company_box)
        company_form.setSpacing(10)

        self._display_name = QLineEdit()
        self._display_name.setPlaceholderText("e.g. ABC Ceramic")
        company_form.addRow(
            "Profile Name",
            self._wrap_validated_field(
                self._display_name, validate_profile_name, "display_name"
            ),
        )

        self._company_name = QLineEdit()
        company_form.addRow("Company Name", self._company_name)

        self._logo_path = QLineEdit()
        self._logo_path.setPlaceholderText("JPG, PNG, JPEG, or WEBP")
        company_form.addRow(
            "Company Logo",
            self._wrap_validated_field(self._logo_path, validate_logo_path, "logo"),
        )

        self._email = QLineEdit()
        self._email.setPlaceholderText("name@company.com")
        company_form.addRow(
            "Email",
            self._wrap_validated_field(self._email, validate_email, "email"),
        )

        self._phone = QLineEdit()
        self._phone.setPlaceholderText("10-digit mobile number")
        self._phone.setMaxLength(10)
        self._phone.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"^\d*$"))
        )
        company_form.addRow(
            "Phone",
            self._wrap_validated_field(self._phone, validate_phone, "phone"),
        )

        self._website = QLineEdit()
        self._website.setPlaceholderText("company.com")
        company_form.addRow(
            "Website",
            self._wrap_validated_field(self._website, validate_website, "website"),
        )

        self._address = QLineEdit()
        company_form.addRow("Address", self._address)

        pdf_row = QHBoxLayout()
        self._pdf_folder = QLineEdit()
        self._pdf_folder.setPlaceholderText("Default folder for exported PDFs")
        pdf_browse = QPushButton("Browse…")
        pdf_browse.setObjectName("BrowseButton")
        pdf_browse.clicked.connect(self._browse_pdf_folder)
        pdf_row.addWidget(self._pdf_folder, stretch=1)
        pdf_row.addWidget(pdf_browse)
        company_form.addRow("Default PDF Folder", pdf_row)

        editor_layout.addWidget(company_box)

        export_box = QGroupBox("Export Options")
        export_box.setObjectName("SettingsSection")
        export_form = QFormLayout(export_box)
        export_form.setSpacing(10)

        self._include_search_image = QCheckBox("Include search image in PDF")
        self._include_search_image.setChecked(True)
        export_form.addRow(self._include_search_image)

        self._include_image_path = QCheckBox("Include image file path in PDF")
        export_form.addRow(self._include_image_path)

        self._export_only_selected = QCheckBox("Export only selected search results")
        export_form.addRow(self._export_only_selected)

        self._watermark = QLineEdit()
        self._watermark.setPlaceholderText("Optional watermark text")
        export_form.addRow("Watermark", self._watermark)

        self._max_results = QSpinBox()
        self._max_results.setRange(1, 100)
        self._max_results.setValue(12)
        export_form.addRow("Max results", self._max_results)

        editor_layout.addWidget(export_box)
        editor_layout.addStretch()

        save_row = QHBoxLayout()
        save_row.addStretch()
        self._save_btn = QPushButton("Save Profile")
        self._save_btn.setObjectName("PrimaryButton")
        self._save_btn.setMinimumWidth(140)
        self._save_btn.clicked.connect(self._on_save)
        save_row.addWidget(self._save_btn)
        editor_layout.addLayout(save_row)

        scroll.setWidget(editor)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        outer.addWidget(splitter, stretch=1)

    def _wrap_validated_field(
        self,
        field: QLineEdit,
        validator: Callable[[str], Optional[str]],
        key: str,
    ) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        if key == "logo":
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(field, stretch=1)
            browse = QPushButton("Browse…")
            browse.setObjectName("BrowseButton")
            browse.clicked.connect(self._browse_logo)
            row.addWidget(browse)
            layout.addLayout(row)
        else:
            layout.addWidget(field)

        error_label = QLabel("")
        error_label.setObjectName("FieldError")
        error_label.setWordWrap(True)
        error_label.hide()
        layout.addWidget(error_label)
        self._field_errors[key] = error_label

        field.editingFinished.connect(
            lambda f=field, v=validator, k=key: self._validate_field(f, v, k)
        )
        return container

    def _set_field_error(
        self,
        field: QLineEdit,
        key: str,
        message: Optional[str],
    ) -> bool:
        error_label = self._field_errors[key]
        if message:
            error_label.setText(message)
            error_label.show()
            field.setProperty("invalid", True)
        else:
            error_label.clear()
            error_label.hide()
            field.setProperty("invalid", False)
        field.style().unpolish(field)
        field.style().polish(field)
        return message is None

    def _validate_field(
        self,
        field: QLineEdit,
        validator: Callable[[str], Optional[str]],
        key: str,
    ) -> bool:
        return self._set_field_error(field, key, validator(field.text()))

    def _clear_validation_hints(self) -> None:
        for label in self._field_errors.values():
            label.clear()
            label.hide()
        for field in (
            self._display_name,
            self._logo_path,
            self._email,
            self._phone,
            self._website,
        ):
            field.setProperty("invalid", False)
            field.style().unpolish(field)
            field.style().polish(field)

    def _validate_all_fields(self) -> bool:
        checks = (
            (self._display_name, validate_profile_name, "display_name"),
            (self._email, validate_email, "email"),
            (self._phone, validate_phone, "phone"),
            (self._website, validate_website, "website"),
            (self._logo_path, validate_logo_path, "logo"),
        )
        results = [
            self._validate_field(field, validator, key)
            for field, validator, key in checks
        ]
        if not all(results):
            for field, _, _ in checks:
                if field.property("invalid"):
                    field.setFocus()
                    break
        return all(results)

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
        self._clear_validation_hints()
        self._display_name.setText(master.display_name)
        self._company_name.setText(master.company_name)
        self._logo_path.setText(master.logo_path)
        self._email.setText(master.email)
        self._phone.setText(master.phone)
        self._website.setText(master.website)
        self._address.setText(master.address)
        self._pdf_folder.setText(master.default_pdf_folder)
        self._include_search_image.setChecked(master.include_search_image)
        self._include_image_path.setChecked(master.include_image_path)
        self._export_only_selected.setChecked(master.export_only_selected)
        self._watermark.setText(master.watermark_text)
        self._max_results.setValue(master.max_results)

    def _clear_form(self) -> None:
        self._editing_id = None
        self._clear_validation_hints()
        for field in (
            self._display_name,
            self._company_name,
            self._logo_path,
            self._email,
            self._phone,
            self._website,
            self._address,
            self._pdf_folder,
            self._watermark,
        ):
            field.clear()
        self._include_search_image.setChecked(True)
        self._include_image_path.setChecked(False)
        self._export_only_selected.setChecked(False)
        self._max_results.setValue(12)
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
            include_search_image=self._include_search_image.isChecked(),
            include_image_path=self._include_image_path.isChecked(),
            export_only_selected=self._export_only_selected.isChecked(),
            watermark_text=self._watermark.text().strip(),
            max_results=self._max_results.value(),
        )
        if self._editing_id:
            return CatalogueMaster(id=self._editing_id, **common)
        return CatalogueMaster(**common)

    def _on_new(self) -> None:
        self._clear_form()
        self._display_name.setFocus()

    def _on_save(self) -> None:
        if not self._validate_all_fields():
            return

        try:
            master = self._form_master()
            if self._editing_id and self._service.get(self._editing_id):
                saved = self._service.update(master)
            else:
                saved = self._service.add(master)
                self._editing_id = saved.id
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
            "Images (*.png *.jpg *.jpeg *.webp)",
        )
        if path:
            self._logo_path.setText(path)
            self._validate_field(self._logo_path, validate_logo_path, "logo")

    def _browse_pdf_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Default PDF Folder")
        if folder:
            self._pdf_folder.setText(folder)

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        self._apply_styles()

    def _apply_styles(self) -> None:
        self.setStyleSheet(get_shared_view_qss(self._theme) + get_settings_view_qss(self._theme))
