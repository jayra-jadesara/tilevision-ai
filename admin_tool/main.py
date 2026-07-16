"""
TileVision AI — Admin License Manager (Vendor Tool).

Run: python admin_tool/main.py
"""

import base64
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PrivateFormat,
    PublicFormat,
    NoEncryption,
    load_pem_private_key,
)

from PySide6.QtCore import Qt, QDate
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCalendarWidget,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QCheckBox,
    QGridLayout,
)

from license_ledger import LicenseLedger, LicenseRecord
from admin_theme import get_admin_qss
from src.licensing.validator import (
    VENDOR_LICENSE_TYPES,
    LIFETIME_EXPIRY_SENTINEL,
    compute_expiry_date,
    generate_license_key,
)
from src.utils.brand_assets import APP_ICON_PATH, logo_pixmap

_SETTINGS_PATH = Path.home() / ".tilevision_ai_vendor" / "admin_settings.json"


class AdminLicenseWindow(QMainWindow):
    """Vendor desktop tool for license generation and customer registry."""

    def __init__(self) -> None:
        super().__init__()
        self._private_key_pem: Optional[bytes] = None
        self._ledger = LicenseLedger()
        self._renew_from_id: Optional[str] = None
        self._current_theme = "light"

        self.setWindowTitle("TileVision AI — Vendor License Manager")
        self.resize(1020, 780)
        if APP_ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self._load_settings()
        self._setup_ui()
        self._apply_styles()
        self._load_saved_private_key()
        self._refresh_all()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        header.setSpacing(12)

        logo_label = QLabel()
        logo_label.setObjectName("BrandLabel")
        logo_scaled = logo_pixmap(44)
        if not logo_scaled.isNull():
            logo_label.setPixmap(logo_scaled)
        header.addWidget(logo_label)

        title_block = QVBoxLayout()
        title_block.setSpacing(2)
        title = QLabel("TileVision AI")
        title.setObjectName("Title")
        subtitle = QLabel("Vendor License Manager")
        subtitle.setObjectName("Subtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block)
        header.addStretch()

        theme_label = QLabel("Theme")
        theme_label.setObjectName("ThemeLabel")
        header.addWidget(theme_label)
        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["light", "dark"])
        idx = self._theme_combo.findText(self._current_theme)
        self._theme_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._theme_combo.setFixedWidth(110)
        self._prepare_form_field(self._theme_combo)
        self._theme_combo.currentTextChanged.connect(self._on_theme_changed)
        header.addWidget(self._theme_combo)
        layout.addLayout(header)

        warning = QLabel(
            "Vendor tool only. Tracks keys YOU issue here. "
            "Built-in 15-day auto-trials on customer PCs are offline and not listed "
            "(no internet). Cancellation blocks new activations; ship revocation updates "
            "to enforce refunds on already-activated PCs."
        )
        warning.setObjectName("Warning")
        warning.setWordWrap(True)
        layout.addWidget(warning)

        layout.addWidget(self._build_keypair_section())

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_overview_tab(), "Overview")
        self._tabs.addTab(self._build_generate_tab(), "Generate Key")
        self._tabs.addTab(self._build_registry_tab(), "Customers & Licenses")
        layout.addWidget(self._tabs, stretch=1)

    def _build_keypair_section(self) -> QGroupBox:
        box = QGroupBox("Signing Key (one-time setup)")
        row = QHBoxLayout(box)
        self._keypair_status = QLabel("No private key loaded.")
        row.addWidget(self._keypair_status, stretch=1)
        load_btn = QPushButton("Load Private Key")
        load_btn.clicked.connect(self._on_load_private_key)
        row.addWidget(load_btn)
        gen_btn = QPushButton("Generate New Keypair")
        gen_btn.clicked.connect(self._on_generate_keypair)
        row.addWidget(gen_btn)
        return box

    def _build_overview_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        grid = QGridLayout()
        self._stat_total = QLabel("0")
        self._stat_active = QLabel("0")
        self._stat_trials = QLabel("0")
        self._stat_official = QLabel("0")
        self._stat_cancelled = QLabel("0")
        self._stat_expiring = QLabel("0")

        for idx, (title, widget) in enumerate(
            [
                ("Total Issued", self._stat_total),
                ("Active", self._stat_active),
                ("Active Trials", self._stat_trials),
                ("Active Official", self._stat_official),
                ("Cancelled", self._stat_cancelled),
                ("Expiring in 30 Days", self._stat_expiring),
            ]
        ):
            card = QGroupBox(title)
            card_layout = QVBoxLayout(card)
            widget.setObjectName("StatValue")
            widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(widget)
            grid.addWidget(card, idx // 3, idx % 3)
        layout.addLayout(grid)

        steps = QLabel(
            "Quick workflow:\n"
            "1. Customer sends Machine ID from TileVision Activation screen\n"
            "2. Generate Key tab → create license → copy to customer\n"
            "3. Customers & Licenses tab → track, renew, or cancel\n"
            "4. Export Revocation List before each app release"
        )
        steps.setWordWrap(True)
        steps.setObjectName("Hint")
        layout.addWidget(steps)
        layout.addStretch()
        return page

    def _build_generate_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        form_box = QGroupBox("Customer & License Details")
        form = QFormLayout(form_box)
        form.setSpacing(14)
        form.setVerticalSpacing(14)
        form.setContentsMargins(12, 16, 12, 12)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self._customer_name = QLineEdit()
        self._customer_name.setPlaceholderText("Sunrise Tile Gallery")
        self._prepare_form_field(self._customer_name)
        form.addRow("Customer Name:", self._customer_name)

        self._contact = QLineEdit()
        self._contact.setPlaceholderText("email / phone (optional)")
        self._prepare_form_field(self._contact)
        form.addRow("Contact:", self._contact)

        self._machine_id = QLineEdit()
        self._machine_id.setPlaceholderText("Paste Machine ID from customer's Activation screen")
        self._prepare_form_field(self._machine_id)
        form.addRow("Machine ID:", self._machine_id)

        self._license_type = QComboBox()
        self._license_type.addItems(list(VENDOR_LICENSE_TYPES))
        self._license_type.currentTextChanged.connect(self._on_license_type_changed)
        self._prepare_form_field(self._license_type)
        form.addRow("License Type:", self._license_type)

        self._expiry = QDateEdit()
        self._expiry.setDisplayFormat("yyyy-MM-dd")
        self._expiry.setMinimumDate(QDate.currentDate())
        self._expiry.setCalendarPopup(True)
        self._expiry.setButtonSymbols(QDateEdit.ButtonSymbols.NoButtons)
        self._prepare_form_field(self._expiry)

        self._expiry_calendar_btn = QPushButton("Open Calendar")
        self._expiry_calendar_btn.setMinimumHeight(36)
        self._expiry_calendar_btn.clicked.connect(self._open_expiry_calendar)

        expiry_row = QWidget()
        expiry_layout = QHBoxLayout(expiry_row)
        expiry_layout.setContentsMargins(0, 0, 0, 0)
        expiry_layout.setSpacing(8)
        expiry_layout.addWidget(self._expiry, stretch=1)
        expiry_layout.addWidget(self._expiry_calendar_btn)
        form.addRow("Expiry Date:", expiry_row)
        self._on_license_type_changed(self._license_type.currentText())

        self._notes = QLineEdit()
        self._notes.setPlaceholderText("Invoice #, sales rep, etc.")
        self._prepare_form_field(self._notes)
        form.addRow("Notes:", self._notes)

        self._wildcard = QCheckBox("Any machine (DEV/TEST only)")
        form.addRow("", self._wildcard)

        self._renew_label = QLabel("")
        self._renew_label.setObjectName("Hint")
        form.addRow("", self._renew_label)

        generate_btn = QPushButton("Generate License Key")
        generate_btn.setObjectName("PrimaryButton")
        generate_btn.setMinimumHeight(40)
        generate_btn.clicked.connect(self._on_generate_license)
        form.addRow("", generate_btn)
        layout.addWidget(form_box)

        output_box = QGroupBox("Generated Key — send this to the customer")
        output_layout = QVBoxLayout(output_box)
        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        output_layout.addWidget(self._output)
        copy_btn = QPushButton("Copy Key to Clipboard")
        copy_btn.clicked.connect(self._on_copy_key)
        output_layout.addWidget(copy_btn)
        layout.addWidget(output_box, stretch=1)
        return page

    def _build_registry_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Status:"))
        self._status_filter = QComboBox()
        self._status_filter.addItems(["all", "active", "cancelled", "superseded"])
        self._prepare_form_field(self._status_filter)
        self._status_filter.currentTextChanged.connect(self._refresh_registry_table)
        filter_row.addWidget(self._status_filter)

        filter_row.addWidget(QLabel("Category:"))
        self._category_filter = QComboBox()
        self._category_filter.addItems(["all", "trials", "official"])
        self._prepare_form_field(self._category_filter)
        self._category_filter.currentTextChanged.connect(self._refresh_registry_table)
        filter_row.addWidget(self._category_filter)

        filter_row.addWidget(QLabel("Search:"))
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Customer, contact, machine ID...")
        self._prepare_form_field(self._search_edit)
        self._search_edit.textChanged.connect(self._refresh_registry_table)
        filter_row.addWidget(self._search_edit, stretch=1)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_all)
        filter_row.addWidget(refresh_btn)
        layout.addLayout(filter_row)

        self._registry_table = QTableWidget(0, 8)
        self._registry_table.setHorizontalHeaderLabels(
            ["Customer", "Contact", "Type", "Expires", "Status", "Machine ID", "Issued", "License ID"]
        )
        self._registry_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._registry_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._registry_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._registry_table.doubleClicked.connect(self._on_renew_selected)
        layout.addWidget(self._registry_table, stretch=1)

        action_row = QHBoxLayout()
        renew_btn = QPushButton("Renew / Extend Selected")
        renew_btn.clicked.connect(self._on_renew_selected)
        action_row.addWidget(renew_btn)

        copy_again_btn = QPushButton("Copy Stored Key")
        copy_again_btn.clicked.connect(self._on_copy_stored_key)
        action_row.addWidget(copy_again_btn)

        cancel_btn = QPushButton("Cancel Selected")
        cancel_btn.setObjectName("DangerButton")
        cancel_btn.clicked.connect(self._on_cancel_selected)
        action_row.addWidget(cancel_btn)

        action_row.addStretch()

        export_csv_btn = QPushButton("Export CSV")
        export_csv_btn.clicked.connect(self._on_export_csv)
        action_row.addWidget(export_csv_btn)

        export_rev_btn = QPushButton("Export Revocation JSON")
        export_rev_btn.clicked.connect(self._on_export_revocation)
        action_row.addWidget(export_rev_btn)

        export_py_btn = QPushButton("Copy Revocation Python")
        export_py_btn.clicked.connect(self._on_copy_revocation_python)
        action_row.addWidget(export_py_btn)

        layout.addLayout(action_row)

        self._registry_log = QTextEdit()
        self._registry_log.setReadOnly(True)
        self._registry_log.setFixedHeight(90)
        layout.addWidget(self._registry_log)
        return page

    def _refresh_all(self) -> None:
        stats = self._ledger.get_stats()
        self._stat_total.setText(str(stats.total))
        self._stat_active.setText(str(stats.active))
        self._stat_trials.setText(str(stats.trials_active))
        self._stat_official.setText(str(stats.official_active))
        self._stat_cancelled.setText(str(stats.cancelled))
        self._stat_expiring.setText(str(stats.expiring_soon))
        self._refresh_registry_table()

    def _refresh_registry_table(self) -> None:
        records = self._ledger.list_licenses(
            status_filter=self._status_filter.currentText(),
            category_filter=self._category_filter.currentText(),
            search_text=self._search_edit.text(),
        )
        self._registry_table.setRowCount(len(records))
        for row, rec in enumerate(records):
            self._registry_table.setItem(row, 0, QTableWidgetItem(rec.customer_name))
            self._registry_table.setItem(row, 1, QTableWidgetItem(rec.contact))
            self._registry_table.setItem(row, 2, QTableWidgetItem(rec.license_type))
            self._registry_table.setItem(row, 3, QTableWidgetItem(rec.expires_at))
            self._registry_table.setItem(row, 4, QTableWidgetItem(rec.status))
            machine_short = rec.machine_id if len(rec.machine_id) <= 24 else rec.machine_id[:21] + "..."
            self._registry_table.setItem(row, 5, QTableWidgetItem(machine_short))
            self._registry_table.setItem(row, 6, QTableWidgetItem(rec.issued_at))
            self._registry_table.setItem(row, 7, QTableWidgetItem(rec.license_id))

    def _selected_record(self) -> Optional[LicenseRecord]:
        license_id = self._selected_license_id()
        if not license_id:
            return None
        return self._ledger.get_license(license_id)

    def _selected_license_id(self) -> Optional[str]:
        row = self._registry_table.currentRow()
        if row < 0:
            return None
        item = self._registry_table.item(row, 7)
        return item.text() if item else None

    def _prepare_form_field(self, widget) -> None:
        """Give form rows enough height so styled inputs are not clipped."""
        widget.setMinimumHeight(36)
        if hasattr(widget, "setMinimumWidth"):
            widget.setMinimumWidth(280)

    def _on_license_type_changed(self, license_type: str) -> None:
        expiry_str = compute_expiry_date(license_type)
        self._set_expiry_date(expiry_str)
        is_lifetime = license_type == "Lifetime"
        self._expiry.setEnabled(not is_lifetime)
        self._expiry_calendar_btn.setEnabled(not is_lifetime)
        if is_lifetime:
            self._expiry.setToolTip("Lifetime licenses do not expire.")
        else:
            self._expiry.setToolTip("Adjust the expiry date if needed.")

    def _set_expiry_date(self, expiry_str: str) -> None:
        date = QDate.fromString(expiry_str, "yyyy-MM-dd")
        if date.isValid():
            self._expiry.setDate(date)

    def _selected_expiry_date(self) -> str:
        if self._license_type.currentText() == "Lifetime":
            return LIFETIME_EXPIRY_SENTINEL
        return self._expiry.date().toString("yyyy-MM-dd")

    def _open_expiry_calendar(self) -> None:
        """Open a modal calendar — reliable on Windows when QDateEdit popup fails."""
        if not self._expiry.isEnabled():
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Select Expiry Date")
        dialog.setMinimumWidth(340)
        dialog.setStyleSheet(get_admin_qss(self._current_theme))

        layout = QVBoxLayout(dialog)
        calendar = QCalendarWidget()
        calendar.setGridVisible(True)
        calendar.setMinimumDate(QDate.currentDate())
        calendar.setSelectedDate(self._expiry.date())
        layout.addWidget(calendar)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        calendar.clicked.connect(self._expiry.setDate)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._expiry.setDate(calendar.selectedDate())

    def _save_settings(self, **updates: str) -> None:
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {}
        if _SETTINGS_PATH.exists():
            try:
                data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data.update(updates)
        _SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_settings(self) -> None:
        if not _SETTINGS_PATH.exists():
            return
        try:
            data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
            theme = str(data.get("theme", "light")).lower()
            if theme in {"light", "dark"}:
                self._current_theme = theme
        except Exception:
            pass

    def _on_theme_changed(self, theme: str) -> None:
        if theme not in {"light", "dark"}:
            return
        self._current_theme = theme
        self._save_settings(theme=theme)
        self._apply_styles()

    def _save_private_key_path(self, path: Path) -> None:
        self._save_settings(private_key_path=str(path))

    def _load_saved_private_key(self) -> None:
        if not _SETTINGS_PATH.exists():
            return
        try:
            data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
            path = Path(data.get("private_key_path", ""))
            if path.exists():
                pem = path.read_bytes()
                load_pem_private_key(pem, password=None)
                self._private_key_pem = pem
                self._keypair_status.setText(f"Loaded: {path.name}")
        except Exception:
            pass

    def _on_load_private_key(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Load Private Key", "", "PEM Files (*.pem)")
        if not path_str:
            return
        try:
            pem = Path(path_str).read_bytes()
            load_pem_private_key(pem, password=None)
            self._private_key_pem = pem
            self._keypair_status.setText(f"Loaded: {Path(path_str).name}")
            self._save_private_key_path(Path(path_str))
        except Exception as exc:
            QMessageBox.critical(self, "Invalid Key", str(exc))

    def _on_generate_keypair(self) -> None:
        if QMessageBox.question(
            self,
            "Generate Keypair",
            "Create a new signing keypair? Embed the new PUBLIC key in "
            "src/licensing/validator.py before customer builds accept new keys.",
        ) != QMessageBox.StandardButton.Yes:
            return

        private_key = ec.generate_private_key(ec.SECP256R1())
        private_pem = private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        public_pem = private_key.public_key().public_bytes(
            Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
        )
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Private Key", "tilevision_private_key.pem", "PEM Files (*.pem)"
        )
        if save_path:
            Path(save_path).write_bytes(private_pem)
            self._save_private_key_path(Path(save_path))
        self._private_key_pem = private_pem
        self._keypair_status.setText("New keypair generated.")
        self._output.setPlainText(
            "Paste into src/licensing/validator.py as EMBEDDED_PUBLIC_KEY_PEM:\n\n"
            + public_pem.decode("utf-8")
        )

    def _on_generate_license(self) -> None:
        if not self._private_key_pem:
            QMessageBox.warning(self, "No Key", "Load or generate a private signing key first.")
            return

        customer = self._customer_name.text().strip()
        machine_id = self._machine_id.text().strip()
        license_type = self._license_type.currentText()
        expires_at = self._selected_expiry_date()
        use_wildcard = self._wildcard.isChecked()

        if not customer:
            QMessageBox.warning(self, "Missing", "Customer name is required.")
            return
        if not use_wildcard and not machine_id:
            QMessageBox.warning(self, "Missing", "Machine ID is required (or enable DEV wildcard).")
            return
        if not use_wildcard and self._ledger.is_machine_blocked(machine_id):
            QMessageBox.warning(
                self,
                "Machine Blocked",
                "This Machine ID has a cancelled license. Resolve with the customer first.",
            )
            return

        hardware_hash = "*" if use_wildcard else machine_id
        try:
            key = generate_license_key(
                self._private_key_pem,
                customer,
                expires_at,
                hardware_hash,
                license_type,
            )
            payload = json.loads(base64.b64decode(key + "==").decode("utf-8"))
            license_id = payload["license_id"]
        except Exception as exc:
            QMessageBox.critical(self, "Failed", f"Could not generate key:\n{exc}")
            return

        self._output.setPlainText(key)
        self._ledger.record_issue(
            license_id=license_id,
            customer_name=customer,
            machine_id=hardware_hash if use_wildcard else machine_id,
            license_type=license_type,
            expires_at=expires_at,
            license_key=key,
            notes=self._notes.text().strip(),
            contact=self._contact.text().strip(),
            supersede_license_id=self._renew_from_id,
        )
        self._renew_from_id = None
        self._renew_label.setText("")
        self._refresh_all()
        self._tabs.setCurrentIndex(2)
        QMessageBox.information(
            self,
            "License Generated",
            f"Key created for {customer}. Copy it and send to the customer.",
        )

    def _on_copy_key(self) -> None:
        text = self._output.toPlainText().strip()
        if text:
            QGuiApplication.clipboard().setText(text)

    def _on_renew_selected(self) -> None:
        rec = self._selected_record()
        if rec is None:
            QMessageBox.information(self, "Select Row", "Select a customer license first.")
            return
        self._renew_from_id = rec.license_id
        self._customer_name.setText(rec.customer_name)
        self._contact.setText(rec.contact)
        self._machine_id.setText(rec.machine_id)
        self._notes.setText(rec.notes)
        idx = self._license_type.findText(rec.license_type)
        if idx >= 0:
            self._license_type.setCurrentIndex(idx)
        self._on_license_type_changed(self._license_type.currentText())
        self._renew_label.setText(f"Renewing / extending license {rec.license_id[:8]}...")
        self._tabs.setCurrentIndex(1)

    def _on_copy_stored_key(self) -> None:
        rec = self._selected_record()
        if rec is None:
            QMessageBox.information(self, "Select Row", "Select a license first.")
            return
        if not rec.license_key:
            QMessageBox.warning(
                self,
                "No Stored Key",
                "This record has no stored key (older entry). Generate a renewal instead.",
            )
            return
        QGuiApplication.clipboard().setText(rec.license_key)
        self._output.setPlainText(rec.license_key)
        self._registry_log.append(
            f"[{datetime.now():%H:%M:%S}] Copied stored key for {rec.customer_name}"
        )

    def _on_cancel_selected(self) -> None:
        license_id = self._selected_license_id()
        if not license_id:
            QMessageBox.information(self, "Select Row", "Select a license first.")
            return

        if QMessageBox.question(
            self,
            "Cancel License",
            "Mark this license as cancelled?\n\n"
            "Offline PCs already activated keep working until expiry or until "
            "you ship an app update with the revocation list.",
        ) != QMessageBox.StandardButton.Yes:
            return

        if self._ledger.cancel_license(license_id, reason="Cancelled by vendor"):
            self._registry_log.append(
                f"[{datetime.now():%H:%M:%S}] Cancelled license {license_id}"
            )
            self._refresh_all()
        else:
            QMessageBox.warning(self, "Not Found", "License ID not found.")

    def _on_export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Registry CSV", "tilevision_licenses.csv", "CSV Files (*.csv)"
        )
        if path:
            self._ledger.export_csv(Path(path))
            QMessageBox.information(self, "Exported", f"Saved to {path}")

    def _on_export_revocation(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Revocation List",
            "revoked_licenses.json",
            "JSON Files (*.json)",
        )
        if path:
            self._ledger.export_revocation_manifest(Path(path))
            QMessageBox.information(self, "Exported", f"Revocation list saved to {path}")

    def _on_copy_revocation_python(self) -> None:
        snippet = self._ledger.export_revocation_python_snippet()
        QGuiApplication.clipboard().setText(snippet)
        QMessageBox.information(
            self,
            "Copied",
            "Python snippet copied. Paste into src/licensing/revocation.py "
            "as EMBEDDED_REVOKED_LICENSE_IDS before your next release.",
        )

    def _apply_styles(self) -> None:
        self.setStyleSheet(get_admin_qss(self._current_theme))
        if hasattr(self, "_theme_combo"):
            idx = self._theme_combo.findText(self._current_theme)
            if idx >= 0 and self._theme_combo.currentIndex() != idx:
                self._theme_combo.blockSignals(True)
                self._theme_combo.setCurrentIndex(idx)
                self._theme_combo.blockSignals(False)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("TileVision AI — Vendor License Manager")
    if APP_ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    window = AdminLicenseWindow()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
