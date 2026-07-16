# src/presentation/dialogs/export_catalog_dialog.py

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QFileDialog,
)

from src.services.catalogue_master_service import CatalogueMaster, CatalogueMasterService
from src.services.pdf_export_service import PdfExportOptions
from src.theme.theme_manager import get_palette, get_shared_view_qss


class ExportCatalogDialog(QDialog):
    """Export search results to PDF — one dialog, profiles managed in Settings."""

    _MANUAL_ENTRY_ID = "__manual__"

    def __init__(
        self,
        parent=None,
        *,
        theme: str = "light",
        on_open_profiles_settings: Optional[Callable[[], None]] = None,
    ) -> None:
        self._default_name = (
            f"tilevision_catalogue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        self._theme = theme
        self._on_open_profiles_settings = on_open_profiles_settings
        self._service = CatalogueMasterService()
        self._output_path: Optional[str] = None
        self._loading_profile = False

        super().__init__(parent)

        self.setWindowTitle("Export Catalogue")
        self.setModal(True)
        self.resize(560, 620)

        self._setup_ui()
        self._reload_profiles()
        self._apply_styles()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Company Profile"))
        self._profile_combo = QComboBox()
        self._profile_combo.currentIndexChanged.connect(self._on_profile_selected)
        profile_row.addWidget(self._profile_combo, stretch=1)
        layout.addLayout(profile_row)

        self._empty_hint = QLabel(
            "No profiles saved yet. Open Settings → Export Profiles to add one."
        )
        self._empty_hint.setWordWrap(True)
        self._empty_hint.setObjectName("PageSubtitle")
        layout.addWidget(self._empty_hint)

        settings_link_row = QHBoxLayout()
        self._settings_link = QPushButton("Open Settings → Export Profiles")
        self._settings_link.setObjectName("SecondaryButton")
        self._settings_link.clicked.connect(self._go_to_settings_profiles)
        settings_link_row.addWidget(self._settings_link)
        settings_link_row.addStretch()
        layout.addLayout(settings_link_row)

        layout.addWidget(QLabel("Company Name"))
        self.company_name = QLineEdit()
        layout.addWidget(self.company_name)

        layout.addWidget(QLabel("Company Logo"))
        logo_row = QHBoxLayout()
        self.logo_edit = QLineEdit()
        logo_row.addWidget(self.logo_edit)
        logo_btn = QPushButton("Browse")
        logo_btn.setObjectName("SecondaryButton")
        logo_btn.clicked.connect(self._browse_logo)
        logo_row.addWidget(logo_btn)
        layout.addLayout(logo_row)

        layout.addWidget(QLabel("Email"))
        self.email_edit = QLineEdit()
        layout.addWidget(self.email_edit)

        layout.addWidget(QLabel("Phone"))
        self.phone_edit = QLineEdit()
        layout.addWidget(self.phone_edit)

        layout.addWidget(QLabel("Website"))
        self.website_edit = QLineEdit()
        layout.addWidget(self.website_edit)

        layout.addWidget(QLabel("Address"))
        self.address_edit = QLineEdit()
        layout.addWidget(self.address_edit)

        layout.addWidget(QLabel("Generated Date & Time"))
        self.datetime_edit = QLineEdit()
        self.datetime_edit.setReadOnly(True)
        self.datetime_edit.setText(datetime.now().strftime("%d %B %Y %I:%M %p"))
        layout.addWidget(self.datetime_edit)

        layout.addWidget(QLabel("Output PDF Path"))
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Choose output PDF path...")
        self.path_edit.setReadOnly(True)
        path_row.addWidget(self.path_edit, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.setObjectName("SecondaryButton")
        browse_btn.clicked.connect(lambda: self._browse_pdf(self._default_name))
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        self.include_search_image = QCheckBox("Include search image")
        self.include_search_image.setChecked(True)
        layout.addWidget(self.include_search_image)

        self.include_image_path = QCheckBox("Include image path")
        self.include_image_path.setChecked(False)
        layout.addWidget(self.include_image_path)

        self.only_selected = QCheckBox("Export only selected results")
        self.only_selected.setChecked(False)
        layout.addWidget(self.only_selected)

        layout.addWidget(QLabel("Watermark"))
        self.watermark_edit = QLineEdit()
        self.watermark_edit.setPlaceholderText("Optional watermark text")
        layout.addWidget(self.watermark_edit)

        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max results"))
        self.max_spin = QSpinBox()
        self.max_spin.setRange(1, 100)
        self.max_spin.setValue(12)
        max_row.addWidget(self.max_spin)
        max_row.addStretch()
        layout.addLayout(max_row)

        button_row = QHBoxLayout()
        button_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("SecondaryButton")
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("Export")
        ok_btn.setObjectName("PrimaryButton")
        ok_btn.clicked.connect(self.accept)
        button_row.addWidget(cancel_btn)
        button_row.addWidget(ok_btn)
        layout.addLayout(button_row)

    def _go_to_settings_profiles(self) -> None:
        if self._on_open_profiles_settings is not None:
            self._on_open_profiles_settings()
        self.reject()

    def _reload_profiles(self) -> None:
        self._loading_profile = True
        self._profile_combo.blockSignals(True)
        self._profile_combo.clear()

        has_profiles = bool(self._service.masters)
        self._empty_hint.setVisible(not has_profiles)

        if has_profiles:
            self._profile_combo.addItem("— Select company profile —", "")
            for master in sorted(self._service.masters, key=lambda m: m.display_name.lower()):
                self._profile_combo.addItem(master.display_name, master.id)

            selected_id = self._service.last_selected_id
            if selected_id:
                index = self._profile_combo.findData(selected_id)
                if index >= 0:
                    self._profile_combo.setCurrentIndex(index)
                    self._apply_master(self._service.get(selected_id))
                else:
                    self._profile_combo.setCurrentIndex(0)
            else:
                self._profile_combo.setCurrentIndex(0)
        else:
            self._profile_combo.addItem("— No saved profiles —", self._MANUAL_ENTRY_ID)
            self._suggest_default_pdf_path(None)

        self._profile_combo.blockSignals(False)
        self._loading_profile = False

    def _on_profile_selected(self, _index: int) -> None:
        if self._loading_profile:
            return

        master_id = self._profile_combo.currentData()
        if not master_id or master_id == self._MANUAL_ENTRY_ID:
            return

        master = self._service.get(str(master_id))
        if master is None:
            return

        self._apply_master(master)
        self._service.set_last_selected(master.id)

    def _apply_master(self, master: Optional[CatalogueMaster]) -> None:
        if master is None:
            return

        self.company_name.setText(master.company_name)
        self.logo_edit.setText(master.logo_path)
        self.email_edit.setText(master.email)
        self.phone_edit.setText(master.phone)
        self.website_edit.setText(master.website)
        self.address_edit.setText(master.address)
        self._suggest_default_pdf_path(master)

    def _suggest_default_pdf_path(self, master: Optional[CatalogueMaster]) -> None:
        if master is not None:
            suggested = self._service.suggested_pdf_path(master, self._default_name)
        else:
            suggested = str(Path.home() / self._default_name)
        self._output_path = suggested
        self.path_edit.setText(suggested)

    def _browse_logo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Company Logo",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if path:
            self.logo_edit.setText(path)

    def _browse_pdf(self, default_name: str) -> None:
        start_dir = str(Path.home())
        if self._output_path:
            start_dir = str(Path(self._output_path).parent)

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PDF",
            str(Path(start_dir) / default_name),
            "PDF Files (*.pdf)",
        )
        if path:
            if not path.lower().endswith(".pdf"):
                path += ".pdf"
            self._output_path = path
            self.path_edit.setText(path)

    def output_path(self) -> Optional[str]:
        return self._output_path

    def selected_master_id(self) -> Optional[str]:
        master_id = self._profile_combo.currentData()
        if not master_id or master_id in ("", self._MANUAL_ENTRY_ID):
            return None
        return str(master_id)

    def options(self) -> PdfExportOptions:
        master_id = self.selected_master_id()
        if master_id and self._output_path:
            self._service.remember_pdf_folder(master_id, self._output_path)

        return PdfExportOptions(
            company_name=self.company_name.text(),
            company_email=self.email_edit.text(),
            company_phone=self.phone_edit.text(),
            company_website=self.website_edit.text(),
            company_address=self.address_edit.text(),
            logo_path=self.logo_edit.text(),
            include_search_image=self.include_search_image.isChecked(),
            include_image_path=self.include_image_path.isChecked(),
            include_selected_only=self.only_selected.isChecked(),
            max_results=self.max_spin.value(),
            watermark_text=self.watermark_edit.text().strip() or None,
        )

    def _apply_styles(self) -> None:
        p = get_palette(self._theme)
        self.setStyleSheet(
            get_shared_view_qss(self._theme)
            + f"""
            QDialog {{ background-color: {p['bg_app']}; }}
            QLabel {{ color: {p['text_secondary']}; }}
            QLineEdit, QSpinBox, QComboBox {{
                background-color: {p['bg_input']};
                border: 1px solid {p['border_strong']};
                border-radius: 6px;
                padding: 6px 8px;
                color: {p['text_primary']};
            }}
            QCheckBox {{ color: {p['text_secondary']}; }}
            """
        )
