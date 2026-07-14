# src/presentation/dialogs/export_catalog_dialog.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QFileDialog,
    QSpinBox,
    QLineEdit,
)

from datetime import datetime
from src.services.pdf_export_service import PdfExportOptions
from src.services.company_settings_service import CompanySettingsService

class ExportCatalogDialog(QDialog):
    def __init__(self, parent=None) -> None:
        default_name = (
            f"tilevision_catalogue_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )

        super().__init__(parent)

        self.setWindowTitle("Export Catalogue")
        self.setModal(True)
        self.resize(520, 320)

        self._output_path: Optional[str] = None

        self._setup_ui(default_name)

    def _setup_ui(self, default_name: str) -> None:
        button_row = QHBoxLayout()
        layout = QVBoxLayout(self)

        settings = CompanySettingsService.load()

        layout.addWidget(QLabel("Company Name"))
        self.company_name = QLineEdit(settings["company_name"])
        layout.addWidget(self.company_name)

        layout.addWidget(QLabel("Company Logo"))

        logo_row = QHBoxLayout()

        self.logo_edit = QLineEdit(settings["logo_path"])
        logo_row.addWidget(self.logo_edit)

        logo_btn = QPushButton("Browse")
        logo_row.addWidget(logo_btn)

        layout.addLayout(logo_row)

        layout.addWidget(QLabel("Email"))
        self.email_edit = QLineEdit(settings["email"])
        layout.addWidget(self.email_edit)

        layout.addWidget(QLabel("Phone"))
        self.phone_edit = QLineEdit(settings["phone"])
        layout.addWidget(self.phone_edit)

        layout.addWidget(QLabel("Website"))
        self.website_edit = QLineEdit(settings["website"])
        layout.addWidget(self.website_edit)

        layout.addWidget(QLabel("Address"))
        self.address_edit = QLineEdit(settings["address"])
        layout.addWidget(self.address_edit)

        layout.addWidget(QLabel("Generated Date & Time"))

        self.datetime_edit = QLineEdit()
        self.datetime_edit.setReadOnly(True)
        self.datetime_edit.setText(
            datetime.now().strftime("%d %B %Y %I:%M %p")
        )

        layout.addWidget(self.datetime_edit)


        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Choose output PDF path...")
        self.path_edit.setReadOnly(True)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(lambda: self._browse(default_name))

        logo_btn.clicked.connect(self._browse_logo)

        row = QHBoxLayout()
        row.addWidget(self.path_edit, 1)
        row.addWidget(browse_btn)
        layout.addLayout(row)

        self.include_search_image = QCheckBox("Include search image")
        self.include_search_image.setChecked(True)
        layout.addWidget(self.include_search_image)

        self.include_image_path = QCheckBox("Include image path")
        self.include_image_path.setChecked(False)
        layout.addWidget(self.include_image_path)

        self.only_selected = QCheckBox("Export only selected results")
        self.only_selected.setChecked(False)
        layout.addWidget(self.only_selected)

        self.watermark_edit = QLineEdit()
        self.watermark_edit.setPlaceholderText("Optional watermark text")
        layout.addWidget(QLabel("Watermark"))
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
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("Export")
        ok_btn.clicked.connect(self.accept)

        button_row.addWidget(cancel_btn)
        button_row.addWidget(ok_btn)
        layout.addLayout(button_row)

    def _browse_logo(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Company Logo",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if path:
            self.logo_edit.setText(path)
            
    def _browse(self, default_name: str) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PDF",
            str(Path.home() / default_name),
            "PDF Files (*.pdf)",
        )
        if path:
            if not path.lower().endswith(".pdf"):
                path += ".pdf"
            self._output_path = path
            self.path_edit.setText(path)

    def output_path(self) -> Optional[str]:
        return self._output_path

    def options(self) -> PdfExportOptions:
        CompanySettingsService.save(
            {
                "company_name": self.company_name.text(),
                "logo_path": self.logo_edit.text(),
                "email": self.email_edit.text(),
                "phone": self.phone_edit.text(),
                "website": self.website_edit.text(),
                "address": self.address_edit.text(),
            }
        )

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
    