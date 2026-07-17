"""
TileVision AI — Admin License Manager (Vendor Tool).

Run: python admin_tool/main.py
"""

import base64
import json
import shutil
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

from PySide6.QtCore import Qt, QDate, QTimer
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
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
    QFrame,
    QGridLayout,
    QScrollArea,
)

from license_ledger import LicenseLedger, LicenseRecord, is_trial_license_type
from admin_theme import get_admin_qss
from vendor_backup import get_last_backup_summary, resolve_backup_dir, run_vendor_backup
from web_date_picker import WebDatePicker
from src.licensing.validator import (
    VENDOR_LICENSE_TYPES,
    LIFETIME_EXPIRY_SENTINEL,
    compute_expiry_date,
    generate_license_key,
)
from src.utils.brand_assets import APP_ICON_PATH, logo_pixmap
from src.theme.theme_manager import get_palette

_VENDOR_DIR = Path.home() / ".tilevision_ai_vendor"
_SETTINGS_PATH = _VENDOR_DIR / "admin_settings.json"
_VENDOR_KEY_PATH = _VENDOR_DIR / "vendor_private_key.pem"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEV_KEY_PATH = _PROJECT_ROOT / "dev_tools" / "dev_private_key.pem"


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
        self._auto_load_signing_key()
        self._refresh_backup_status()
        self._trigger_vendor_backup(silent=True)
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
            "Trial and full licenses both require a key generated here — "
            "choose a Trial type (e.g. 15-Day Trial) when issuing trial keys. "
            "Cancellation blocks new activations; ship revocation updates "
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
        box = QGroupBox("Signing Key")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        self._keypair_status = QLabel("No signing key loaded.")
        self._keypair_status.setObjectName("KeyStatus")
        layout.addWidget(self._keypair_status)

        hint = QLabel(
            f"Signing key (fixed): {_VENDOR_KEY_PATH}\n"
            "Automatic backup copies your vendor folder to OneDrive/Documents "
            "(cloud-synced) whenever you open the tool or issue a license."
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._backup_status = QLabel(get_last_backup_summary())
        self._backup_status.setObjectName("Hint")
        self._backup_status.setWordWrap(True)
        layout.addWidget(self._backup_status)

        row = QHBoxLayout()
        open_btn = QPushButton("Import Key File...")
        open_btn.setObjectName("PrimaryButton")
        open_btn.setToolTip("Copy a .pem file into your vendor folder and load it")
        open_btn.clicked.connect(self._on_load_private_key)
        row.addWidget(open_btn)

        create_btn = QPushButton("Create New Key (First Setup)")
        create_btn.setToolTip("Only when starting fresh for real customers — not for daily use")
        create_btn.clicked.connect(self._on_generate_keypair)
        row.addWidget(create_btn)

        backup_btn = QPushButton("Backup Now")
        backup_btn.setToolTip("Copy vendor data to OneDrive/Documents now")
        backup_btn.clicked.connect(self._on_backup_now)
        row.addWidget(backup_btn)
        row.addStretch()
        layout.addLayout(row)
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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        form_box = QGroupBox("Customer & License Details")
        form = QFormLayout(form_box)
        form.setSpacing(14)
        form.setVerticalSpacing(18)
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
        self._license_type.activated.connect(self._on_license_type_activated)
        self._prepare_form_field(self._license_type)
        form.addRow("License Type:", self._license_type)

        self._expiry = WebDatePicker(get_qss=lambda: get_admin_qss(self._current_theme))
        self._expiry.setDisplayFormat("yyyy-MM-dd")
        self._expiry.setMinimumDate(QDate.currentDate())
        self._prepare_form_field(self._expiry)

        expiry_row = QWidget()
        expiry_row.setObjectName("ExpiryDateRow")
        expiry_row.setFixedHeight(40)
        expiry_row_layout = QHBoxLayout(expiry_row)
        expiry_row_layout.setContentsMargins(0, 2, 0, 2)
        expiry_row_layout.setSpacing(0)
        expiry_row_layout.addWidget(self._expiry, alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        expiry_row_layout.addStretch()
        form.addRow("Expiry Date:", expiry_row)

        self._expiry_hint = QLabel("")
        self._expiry_hint.setObjectName("Hint")
        form.addRow("", self._expiry_hint)
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
        generate_btn.setMinimumHeight(44)
        generate_btn.clicked.connect(self._on_generate_license)
        form.addRow("", generate_btn)
        layout.addWidget(form_box)

        output_box = QGroupBox("Generated Key — send this to the customer")
        output_layout = QVBoxLayout(output_box)
        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setMinimumHeight(72)
        self._output.setMaximumHeight(120)
        output_layout.addWidget(self._output)
        self._copy_key_btn = QPushButton("Copy Key to Clipboard")
        self._copy_key_btn.clicked.connect(self._on_copy_key)
        output_layout.addWidget(self._copy_key_btn)
        self._copy_key_status = QLabel("")
        self._copy_key_status.setObjectName("CopyFeedback")
        output_layout.addWidget(self._copy_key_status)
        layout.addWidget(output_box)
        layout.addStretch()

        scroll.setWidget(page)
        return scroll

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

        legend = QLabel("")
        legend.setObjectName("Hint")
        legend.setTextFormat(Qt.TextFormat.RichText)
        self._registry_legend = legend
        self._update_status_legend()
        layout.addWidget(legend)

        action_row = QHBoxLayout()
        renew_btn = QPushButton("Renew / Extend Selected")
        renew_btn.clicked.connect(self._on_renew_selected)
        action_row.addWidget(renew_btn)

        copy_again_btn = QPushButton("Copy Stored Key")
        copy_again_btn.clicked.connect(self._on_copy_stored_key)
        self._copy_stored_key_btn = copy_again_btn
        action_row.addWidget(copy_again_btn)

        cancel_btn = QPushButton("Cancel Selected")
        cancel_btn.setObjectName("DangerButton")
        cancel_btn.clicked.connect(self._on_cancel_selected)
        action_row.addWidget(cancel_btn)

        reissue_btn = QPushButton("Allow Re-issue")
        reissue_btn.setToolTip(
            "Clear the block on a Machine ID after a cancelled license, "
            "so you can generate a new key for that PC."
        )
        reissue_btn.clicked.connect(self._on_allow_reissue)
        action_row.addWidget(reissue_btn)

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
            self._registry_table.setItem(row, 4, self._make_status_table_item(rec))
            machine_short = rec.machine_id if len(rec.machine_id) <= 24 else rec.machine_id[:21] + "..."
            self._registry_table.setItem(row, 5, QTableWidgetItem(machine_short))
            self._registry_table.setItem(row, 6, QTableWidgetItem(rec.issued_at))
            self._registry_table.setItem(row, 7, QTableWidgetItem(rec.license_id))

    def _make_status_table_item(self, rec: LicenseRecord) -> QTableWidgetItem:
        """Status column with color coding: green active, yellow trial, red suspended."""
        p = get_palette(self._current_theme)
        status = rec.status.lower()

        if status in ("cancelled", "superseded"):
            label = "Suspended" if status == "cancelled" else "Superseded"
            bg, fg = p["danger_bg"], p["danger_text"]
        elif status == "active" and is_trial_license_type(rec.license_type):
            label = "Trial"
            bg, fg = p["warning_bg"], p["warning_text"]
        elif status == "active":
            label = "Active"
            bg, fg = p["success_bg"], p["success_text"]
        else:
            label = rec.status
            bg, fg = p["bg_panel"], p["text_primary"]

        item = QTableWidgetItem(label)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        item.setBackground(QBrush(QColor(bg)))
        item.setForeground(QBrush(QColor(fg)))
        item.setToolTip(f"Registry status: {rec.status} · {rec.license_type}")
        return item

    def _update_status_legend(self) -> None:
        if not hasattr(self, "_registry_legend"):
            return
        p = get_palette(self._current_theme)
        self._registry_legend.setText(
            "Status column: "
            f"<span style='background:{p['success_bg']};color:{p['success_text']};"
            "padding:2px 8px;border-radius:4px;'>Active</span> "
            f"<span style='background:{p['warning_bg']};color:{p['warning_text']};"
            "padding:2px 8px;border-radius:4px;'>Trial</span> "
            f"<span style='background:{p['danger_bg']};color:{p['danger_text']};"
            "padding:2px 8px;border-radius:4px;'>Suspended</span>"
        )

    def _copy_to_clipboard_with_feedback(
        self,
        text: str,
        *,
        button: Optional[QPushButton] = None,
        status_label: Optional[QLabel] = None,
        empty_title: str = "Nothing to Copy",
        empty_message: str = "Generate a license key first, then copy it.",
        log_message: Optional[str] = None,
    ) -> None:
        cleaned = text.strip()
        if not cleaned:
            QMessageBox.warning(self, empty_title, empty_message)
            if status_label is not None:
                status_label.setText("")
            return

        QGuiApplication.clipboard().setText(cleaned)

        if button is not None:
            original = button.text()
            button.setText("Copied!")
            button.setEnabled(False)

            def _reset_button() -> None:
                button.setText(original)
                button.setEnabled(True)

            QTimer.singleShot(2500, _reset_button)

        if status_label is not None:
            p = get_palette(self._current_theme)
            status_label.setStyleSheet(
                f"color: {p['success_text']}; font-weight: 600; font-size: 11px;"
            )
            status_label.setText("Copied to clipboard.")
            QTimer.singleShot(4000, status_label.clear)

        if log_message:
            self._registry_log.append(f"[{datetime.now():%H:%M:%S}] {log_message}")

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
        if isinstance(widget, WebDatePicker):
            widget.setFixedSize(210, 36)
        elif hasattr(widget, "setMinimumWidth"):
            widget.setMinimumWidth(280)

    def _on_license_type_activated(self, _index: int) -> None:
        self._on_license_type_changed(self._license_type.currentText())

    def _on_license_type_changed(self, license_type: str) -> None:
        expiry_str = compute_expiry_date(license_type)
        self._set_expiry_date(expiry_str)
        is_lifetime = license_type == "Lifetime"
        self._expiry.setEnabled(not is_lifetime)
        if is_lifetime:
            self._expiry.setToolTip("Lifetime licenses do not expire.")
            self._expiry_hint.setText("Auto-set: Lifetime (no expiry)")
        else:
            self._expiry.setToolTip("Adjust the expiry date if needed.")
            self._expiry_hint.setText(f"Auto-set from license type: {expiry_str}")

    def _set_expiry_date(self, expiry_str: str) -> None:
        date = QDate.fromString(expiry_str, "yyyy-MM-dd")
        if date.isValid():
            self._expiry.setDate(date, force=True)

    def _selected_expiry_date(self) -> str:
        if self._license_type.currentText() == "Lifetime":
            return LIFETIME_EXPIRY_SENTINEL
        return self._expiry.date().toString("yyyy-MM-dd")

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

    def _refresh_backup_status(self) -> None:
        if hasattr(self, "_backup_status"):
            backup_dir = resolve_backup_dir()
            base = get_last_backup_summary()
            if backup_dir is not None:
                self._backup_status.setText(
                    f"{base}\nAuto-backup folder: {backup_dir}"
                )
            else:
                self._backup_status.setText(
                    f"{base}\nAuto-backup folder not found — install/sync OneDrive or use Documents."
                )

    def _trigger_vendor_backup(self, *, silent: bool = False) -> None:
        ok, message = run_vendor_backup()
        self._refresh_backup_status()
        if not silent and not ok:
            QMessageBox.warning(self, "Backup", message)
        elif not silent and ok:
            QMessageBox.information(self, "Backup", message)

    def _on_backup_now(self) -> None:
        self._trigger_vendor_backup(silent=False)

    def _ensure_vendor_dir(self) -> None:
        _VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    def _set_key_status(self, loaded: bool, message: str) -> None:
        self._keypair_status.setText(message)
        self._keypair_status.setProperty("loaded", loaded)
        self._keypair_status.style().unpolish(self._keypair_status)
        self._keypair_status.style().polish(self._keypair_status)

    def _save_vendor_private_key(self, pem: bytes) -> None:
        self._ensure_vendor_dir()
        _VENDOR_KEY_PATH.write_bytes(pem)

    def _load_vendor_private_key(self) -> bool:
        if not _VENDOR_KEY_PATH.exists():
            return False
        try:
            pem = _VENDOR_KEY_PATH.read_bytes()
            load_pem_private_key(pem, password=None)
            self._private_key_pem = pem
            self._set_key_status(True, f"Ready — key loaded from {_VENDOR_DIR.name}")
            return True
        except Exception as exc:
            self._private_key_pem = None
            self._set_key_status(False, "Invalid signing key in vendor folder.")
            QMessageBox.critical(self, "Invalid Key", str(exc))
            return False

    def _import_private_key_file(self, source: Path) -> bool:
        try:
            pem = source.read_bytes()
            load_pem_private_key(pem, password=None)
        except Exception as exc:
            self._private_key_pem = None
            self._set_key_status(False, f"Invalid key file: {source.name}")
            QMessageBox.critical(self, "Invalid Key", str(exc))
            return False

        self._save_vendor_private_key(pem)
        self._private_key_pem = pem
        self._set_key_status(True, f"Ready — key saved to {_VENDOR_DIR.name}")
        self._trigger_vendor_backup(silent=True)
        return True

    def _migrate_legacy_key_path(self) -> bool:
        if not _SETTINGS_PATH.exists():
            return False
        try:
            data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
            legacy = Path(data.get("private_key_path", ""))
            if not legacy.exists() or legacy.resolve() == _VENDOR_KEY_PATH.resolve():
                return False
            shutil.copy2(legacy, _VENDOR_KEY_PATH)
            data.pop("private_key_path", None)
            _SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return self._load_vendor_private_key()
        except Exception:
            return False

    def _seed_dev_key_if_needed(self) -> bool:
        if _VENDOR_KEY_PATH.exists() or not _DEV_KEY_PATH.exists():
            return False
        try:
            shutil.copy2(_DEV_KEY_PATH, _VENDOR_KEY_PATH)
            return self._load_vendor_private_key()
        except Exception:
            return False

    def _auto_load_signing_key(self) -> None:
        if self._load_vendor_private_key():
            return
        if self._migrate_legacy_key_path():
            return
        if self._seed_dev_key_if_needed():
            return
        self._set_key_status(
            False,
            f"No signing key — import a .pem or create one. Saved to:\n{_VENDOR_KEY_PATH}",
        )

    def _on_load_private_key(self) -> None:
        start_dir = str(_VENDOR_DIR if _VENDOR_DIR.exists() else Path.home())
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Import Private Key File", start_dir, "PEM Files (*.pem)"
        )
        if not path_str:
            return
        self._import_private_key_file(Path(path_str))

    def _on_generate_keypair(self) -> None:
        if _VENDOR_KEY_PATH.exists():
            overwrite = QMessageBox.question(
                self,
                "Replace Signing Key",
                f"A key already exists at:\n{_VENDOR_KEY_PATH}\n\n"
                "Replace it with a new keypair? Old license keys will stop working "
                "unless the customer app has the matching public key.",
            )
            if overwrite != QMessageBox.StandardButton.Yes:
                return

        if QMessageBox.question(
            self,
            "Create New Signing Key",
            "Use this only for first-time production setup.\n\n"
            "This creates a NEW private/public key pair. You must:\n"
            "1. Key is saved to your vendor folder automatically\n"
            "2. Paste the public key into src/licensing/validator.py\n"
            "3. Rebuild the customer app\n\n"
            "Continue?",
        ) != QMessageBox.StandardButton.Yes:
            return

        private_key = ec.generate_private_key(ec.SECP256R1())
        private_pem = private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        public_pem = private_key.public_key().public_bytes(
            Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
        )
        self._save_vendor_private_key(private_pem)
        self._private_key_pem = private_pem
        self._set_key_status(True, f"Ready — new key saved to {_VENDOR_DIR.name}")
        self._output.setPlainText(
            "Paste into src/licensing/validator.py as EMBEDDED_PUBLIC_KEY_PEM:\n\n"
            + public_pem.decode("utf-8")
        )
        QMessageBox.information(
            self,
            "Key Created",
            f"Private key saved to:\n{_VENDOR_KEY_PATH}\n\n"
            "Public key is shown in the output box below. "
            "Embed it in validator.py before shipping the customer app.",
        )
        self._trigger_vendor_backup(silent=True)

    def _on_generate_license(self) -> None:
        if not self._private_key_pem:
            QMessageBox.warning(
                self,
                "No Signing Key",
                "Open your private key .pem file first (Signing Key section at the top).",
            )
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
            unblock = QMessageBox.question(
                self,
                "Machine Blocked",
                "This Machine ID has a cancelled license.\n\n"
                "Do you want to allow re-issue for this PC now?\n"
                "(Cancelled keys stay on the revocation list.)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if unblock == QMessageBox.StandardButton.Yes:
                self._ledger.unblock_machine(machine_id, reason="Approved during key generation")
                self._registry_log.append(
                    f"[{datetime.now():%H:%M:%S}] Allowed re-issue for {machine_id[:16]}..."
                )
            else:
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
        self._trigger_vendor_backup(silent=True)
        self._tabs.setCurrentIndex(2)
        QMessageBox.information(
            self,
            "License Generated",
            f"Key created for {customer}. Copy it and send to the customer.",
        )

    def _on_copy_key(self) -> None:
        self._copy_to_clipboard_with_feedback(
            self._output.toPlainText(),
            button=self._copy_key_btn,
            status_label=self._copy_key_status,
            empty_message="Generate a license key first, then click Copy.",
        )

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
        self._copy_to_clipboard_with_feedback(
            rec.license_key,
            button=self._copy_stored_key_btn,
            empty_title="No Stored Key",
            empty_message="This record has no stored key (older entry). Generate a renewal instead.",
            log_message=f"Copied stored key for {rec.customer_name}",
        )
        if rec.license_key:
            self._output.setPlainText(rec.license_key)

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

    def _on_allow_reissue(self) -> None:
        rec = self._selected_record()
        if rec is None:
            QMessageBox.information(
                self,
                "Select Row",
                "Select a license row for the customer's Machine ID first.",
            )
            return

        machine_id = rec.machine_id
        if machine_id == "*":
            QMessageBox.information(
                self,
                "Not Applicable",
                "Wildcard DEV keys are not blocked by Machine ID.",
            )
            return

        if not self._ledger.is_machine_blocked(machine_id):
            if self._ledger.is_machine_unblocked(machine_id):
                QMessageBox.information(
                    self,
                    "Already Allowed",
                    "Re-issue is already allowed for this Machine ID.\n\n"
                    "Go to Generate Key and create a new license.",
                )
            else:
                QMessageBox.information(
                    self,
                    "Not Blocked",
                    "This Machine ID has no cancelled license blocking new keys.",
                )
            return

        confirm = QMessageBox.question(
            self,
            "Allow Re-issue",
            f"Allow new license keys for this PC?\n\n"
            f"Customer: {rec.customer_name}\n"
            f"Machine ID: {machine_id}\n\n"
            "Cancelled license(s) remain on the revocation list.\n"
            "Only new key generation is unblocked.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        if self._ledger.unblock_machine(machine_id, reason="Vendor approved re-issue"):
            self._registry_log.append(
                f"[{datetime.now():%H:%M:%S}] Allowed re-issue for {rec.customer_name} "
                f"({machine_id[:16]}...)"
            )
            self._refresh_all()
            QMessageBox.information(
                self,
                "Re-issue Allowed",
                "You can now generate a new key for this Machine ID on the Generate Key tab.",
            )
        else:
            QMessageBox.warning(self, "Failed", "Could not unblock this Machine ID.")

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
        self._update_status_legend()
        if hasattr(self, "_registry_table"):
            self._refresh_registry_table()
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
