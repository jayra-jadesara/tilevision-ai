# src/presentation/dialogs/export_catalog_dialog.py

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QFileDialog,
    QMessageBox,
)

from src.services.catalogue_master_service import CatalogueMaster, CatalogueMasterService
from src.services.pdf_export_service import PdfExportOptions
from src.theme.theme_manager import get_palette, get_shared_view_qss


class ExportCatalogDialog(QDialog):
    """Pick a saved profile, review details, and export — options live in Settings."""

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
        self._current_master: Optional[CatalogueMaster] = None

        super().__init__(parent)

        self.setWindowTitle("Export Catalogue")
        self.setModal(True)
        self.resize(520, 560)

        self._setup_ui()
        self._reload_profiles()
        self._apply_styles()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Company Profile"))
        self._profile_combo = QComboBox()
        self._profile_combo.currentIndexChanged.connect(self._on_profile_selected)
        profile_row.addWidget(self._profile_combo, stretch=1)
        layout.addLayout(profile_row)

        self._empty_hint = QLabel(
            "No profiles yet. Create one under Settings → Export Profiles, then come back here."
        )
        self._empty_hint.setWordWrap(True)
        self._empty_hint.setObjectName("PageSubtitle")
        layout.addWidget(self._empty_hint)

        settings_row = QHBoxLayout()
        self._settings_link = QPushButton("Open Settings → Export Profiles")
        self._settings_link.setObjectName("SecondaryButton")
        self._settings_link.clicked.connect(self._go_to_settings_profiles)
        settings_row.addWidget(self._settings_link)
        settings_row.addStretch()
        layout.addLayout(settings_row)

        company_box = QGroupBox("Company Details")
        company_form = QFormLayout(company_box)
        self.company_name = QLineEdit()
        self.logo_edit = QLineEdit()
        self.email_edit = QLineEdit()
        self.phone_edit = QLineEdit()
        self.website_edit = QLineEdit()
        self.address_edit = QLineEdit()
        for label, field in (
            ("Company Name", self.company_name),
            ("Company Logo", self.logo_edit),
            ("Email", self.email_edit),
            ("Phone", self.phone_edit),
            ("Website", self.website_edit),
            ("Address", self.address_edit),
        ):
            field.setReadOnly(True)
            company_form.addRow(label, field)
        layout.addWidget(company_box)

        options_box = QGroupBox("Export Options")
        options_form = QFormLayout(options_box)
        self._opt_search_image = QLineEdit()
        self._opt_image_path = QLineEdit()
        self._opt_selected_only = QLineEdit()
        self._opt_watermark = QLineEdit()
        self._opt_max_results = QLineEdit()
        for field in (
            self._opt_search_image,
            self._opt_image_path,
            self._opt_selected_only,
            self._opt_watermark,
            self._opt_max_results,
        ):
            field.setReadOnly(True)
        options_form.addRow("Include search image", self._opt_search_image)
        options_form.addRow("Include image path", self._opt_image_path)
        options_form.addRow("Export only selected", self._opt_selected_only)
        options_form.addRow("Watermark", self._opt_watermark)
        options_form.addRow("Max results", self._opt_max_results)
        layout.addWidget(options_box)

        output_box = QGroupBox("Output")
        output_form = QFormLayout(output_box)
        self.datetime_edit = QLineEdit()
        self.datetime_edit.setReadOnly(True)
        self.datetime_edit.setText(datetime.now().strftime("%d %B %Y %I:%M %p"))
        output_form.addRow("Generated", self.datetime_edit)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("Uses profile default PDF folder")
        path_row.addWidget(self.path_edit, stretch=1)
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.setObjectName("BrowseButton")
        self._browse_btn.clicked.connect(lambda: self._browse_pdf(self._default_name))
        path_row.addWidget(self._browse_btn)
        output_form.addRow("PDF file", path_row)
        layout.addWidget(output_box)

        button_row = QHBoxLayout()
        button_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("SecondaryButton")
        cancel_btn.clicked.connect(self.reject)
        self._export_btn = QPushButton("Export Catalogue")
        self._export_btn.setObjectName("PrimaryButton")
        self._export_btn.clicked.connect(self._on_export_clicked)
        button_row.addWidget(cancel_btn)
        button_row.addWidget(self._export_btn)
        layout.addLayout(button_row)

        self._read_only_fields = (
            self.company_name,
            self.logo_edit,
            self.email_edit,
            self.phone_edit,
            self.website_edit,
            self.address_edit,
            self._opt_search_image,
            self._opt_image_path,
            self._opt_selected_only,
            self._opt_watermark,
            self._opt_max_results,
        )

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
        self._settings_link.setVisible(not has_profiles)

        if has_profiles:
            for master in sorted(self._service.masters, key=lambda m: m.display_name.lower()):
                self._profile_combo.addItem(master.display_name, master.id)

            selected_id = self._service.last_selected_id
            index = 0
            if selected_id:
                found = self._profile_combo.findData(selected_id)
                if found >= 0:
                    index = found
            self._profile_combo.setCurrentIndex(index)
            master_id = self._profile_combo.currentData()
            if master_id:
                self._apply_master(self._service.get(str(master_id)))
        else:
            self._profile_combo.addItem("— No saved profiles —", "")
            self._apply_master(None)

        self._profile_combo.blockSignals(False)
        self._loading_profile = False
        self._update_export_enabled()

    def _on_profile_selected(self, _index: int) -> None:
        if self._loading_profile:
            return

        master_id = self._profile_combo.currentData()
        if not master_id:
            self._apply_master(None)
            return

        master = self._service.get(str(master_id))
        if master is None:
            self._apply_master(None)
            return

        self._apply_master(master)
        self._service.set_last_selected(master.id)

    def _apply_master(self, master: Optional[CatalogueMaster]) -> None:
        self._current_master = master
        if master is None:
            for field in self._read_only_fields:
                field.clear()
            self.path_edit.clear()
            self._output_path = None
            self._update_export_enabled()
            return

        self.company_name.setText(master.company_name)
        self.logo_edit.setText(master.logo_path)
        self.email_edit.setText(master.email)
        self.phone_edit.setText(master.phone)
        self.website_edit.setText(master.website)
        self.address_edit.setText(master.address)

        self._opt_search_image.setText("Yes" if master.include_search_image else "No")
        self._opt_image_path.setText("Yes" if master.include_image_path else "No")
        self._opt_selected_only.setText("Yes" if master.export_only_selected else "No")
        self._opt_watermark.setText(master.watermark_text or "—")
        self._opt_max_results.setText(str(master.max_results))

        suggested = self._service.suggested_pdf_path(master, self._default_name)
        self._output_path = suggested
        self.path_edit.setText(suggested)
        self._update_export_enabled()

    def _update_export_enabled(self) -> None:
        has_profile = self._current_master is not None
        self._export_btn.setEnabled(has_profile)
        self._browse_btn.setEnabled(has_profile)
        for field in self._read_only_fields:
            field.setEnabled(has_profile)

    def _on_export_clicked(self) -> None:
        if self._current_master is None:
            QMessageBox.information(
                self,
                "Select Profile",
                "Choose a company profile first, or create one in Settings → Export Profiles.",
            )
            return
        if not self._output_path:
            QMessageBox.warning(self, "Output Path", "Choose where to save the PDF.")
            return
        self.accept()

    def _browse_pdf(self, default_name: str) -> None:
        if self._current_master is None:
            return

        start_dir = str(Path.home())
        if self._output_path:
            start_dir = str(Path(self._output_path).parent)
        elif self._current_master.default_pdf_folder:
            start_dir = self._current_master.default_pdf_folder

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
        if not master_id:
            return None
        return str(master_id)

    def options(self) -> PdfExportOptions:
        master = self._current_master
        if master is None:
            raise RuntimeError("No profile selected.")

        master_id = master.id
        if master_id and self._output_path:
            self._service.remember_pdf_folder(master_id, self._output_path)

        return PdfExportOptions(
            company_name=master.company_name,
            company_email=master.email,
            company_phone=master.phone,
            company_website=master.website,
            company_address=master.address,
            logo_path=master.logo_path,
            include_search_image=master.include_search_image,
            include_image_path=master.include_image_path,
            include_selected_only=master.export_only_selected,
            max_results=master.max_results,
            watermark_text=master.watermark_text.strip() or None,
        )

    def _apply_styles(self) -> None:
        p = get_palette(self._theme)
        self.setStyleSheet(
            get_shared_view_qss(self._theme)
            + f"""
            QDialog {{ background-color: {p['bg_app']}; }}
            QLabel {{ color: {p['text_secondary']}; }}
            QLineEdit, QComboBox {{
                background-color: {p['bg_input']};
                border: 1px solid {p['border_strong']};
                border-radius: 6px;
                padding: 6px 8px;
                color: {p['text_primary']};
            }}
            QLineEdit:read-only {{
                background-color: {p['bg_panel_alt']};
                color: {p['text_secondary']};
            }}
            QGroupBox {{
                color: {p['text_primary']};
                border: 1px solid {p['border']};
                border-radius: 8px;
                margin-top: 12px;
                padding: 10px 10px 6px 10px;
                font-weight: 600;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: {p['accent_text']};
            }}
            """
        )
