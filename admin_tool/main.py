"""
TileVision AI — Admin License Manager.

A standalone PySide6 desktop tool for the VENDOR to generate offline
license keys for customers. This is NOT part of the shipped end-user
application — it embeds/loads the ECDSA PRIVATE key and must never be
distributed to customers or bundled into the customer-facing installer.

Run with:
    python admin_tool/main.py

Capabilities (mapped to the product spec's "Admin can..." list):
    - Generate License Keys — fill in the form, click Generate.
    - Activate License — n/a here; activation happens in the customer's
      copy of the app by pasting the generated key into its License
      Activation dialog.
    - Extend License / Change Expiry Date — generate a new key for the
      same customer + Machine ID with a later expiry date. The customer
      re-activates with the new key, which simply supersedes the old one
      (see IMPORTANT note below on what "offline" implies here).
    - Generate Lifetime License — select "Lifetime" as the license type.
    - Deactivate License / Reset Hardware Binding — see IMPORTANT note.

IMPORTANT — what "offline" means for admin operations:
    Because the end-user application never phones home, there is no way
    for this tool to remotely invalidate a key that's already been
    activated on a customer's machine (true "deactivation" would require
    a server + periodic check-in, which contradicts the "no cloud, no
    internet dependency" requirement). What this tool CAN do:
      - "Extend a license": generate a new key with a later expiry. The
        customer pastes it in; the app's activation flow overwrites the
        previously stored license with the new one.
      - "Reset hardware binding" / move to a new PC: generate a new key
        using the customer's NEW Machine ID. Their OLD key simply won't
        validate on the new machine (hardware mismatch) — nothing needs
        to be explicitly "deactivated" on the old machine for this to work,
        though the old key would, if reused, still validate there.
      - "Deactivate" a compromised/refunded key before its natural expiry
        genuinely cannot be done for an already-offline-activated
        installation. The practical mitigation is keeping a private log of
        issued Machine IDs and refusing to generate further keys for a
        known-bad one going forward.
"""

import base64
import json
import logging
import sys
from datetime import date, datetime
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

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QClipboard, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QPushButton,
    QTextEdit,
    QPlainTextEdit,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QCheckBox,
)

from src.licensing.validator import (
    generate_license_key,
    compute_expiry_date,
    LICENSE_TYPE_DURATIONS_DAYS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tilevision.admin_tool")

# License types an admin can actually issue. "15-Day Trial" is deliberately
# excluded — trials are started automatically client-side (TrialManager),
# never hand-issued as a signed key.
_ISSUABLE_LICENSE_TYPES = ["30-Day", "90-Day", "1-Year", "Lifetime"]


class AdminLicenseWindow(QMainWindow):
    """Main window for the vendor-side License Manager tool."""

    def __init__(self) -> None:
        super().__init__()
        self._private_key_pem: Optional[bytes] = None
        self._private_key_path: Optional[Path] = None

        self.setWindowTitle("TileVision AI — Admin License Manager")
        self.resize(760, 720)
        self._setup_ui()
        self._apply_styles()

    # ── UI Construction ─────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title = QLabel("🔑  TileVision AI — Admin License Manager")
        title.setObjectName("Title")
        layout.addWidget(title)

        warning = QLabel(
            "⚠️ VENDOR TOOL ONLY. Never bundle this tool or your private key "
            "with the customer-facing installer."
        )
        warning.setObjectName("Warning")
        warning.setWordWrap(True)
        layout.addWidget(warning)

        layout.addWidget(self._build_keypair_section())
        layout.addWidget(self._build_generate_section())
        layout.addWidget(self._build_output_section(), stretch=1)

    def _build_keypair_section(self) -> QGroupBox:
        box = QGroupBox("1. Signing Keypair")
        layout = QVBoxLayout(box)

        row = QHBoxLayout()
        self._keypair_status_label = QLabel("No private key loaded.")
        self._keypair_status_label.setObjectName("KeypairStatus")
        row.addWidget(self._keypair_status_label, stretch=1)

        load_button = QPushButton("Load Private Key (.pem)")
        load_button.clicked.connect(self._on_load_private_key)
        row.addWidget(load_button)

        generate_button = QPushButton("Generate New Keypair")
        generate_button.clicked.connect(self._on_generate_keypair)
        row.addWidget(generate_button)

        layout.addLayout(row)
        return box

    def _build_generate_section(self) -> QGroupBox:
        box = QGroupBox("2. Generate a License Key")
        form = QFormLayout(box)

        self._customer_name_edit = QLineEdit()
        self._customer_name_edit.setPlaceholderText("e.g. Sunrise Tile Gallery")
        form.addRow("Customer Name:", self._customer_name_edit)

        self._machine_id_edit = QLineEdit()
        self._machine_id_edit.setPlaceholderText(
            "Paste the customer's Hardware Fingerprint (from their Activation screen)"
        )
        form.addRow("Machine ID:", self._machine_id_edit)

        self._license_type_combo = QComboBox()
        self._license_type_combo.addItems(_ISSUABLE_LICENSE_TYPES)
        self._license_type_combo.currentTextChanged.connect(self._on_license_type_changed)
        form.addRow("License Type:", self._license_type_combo)

        self._expiry_edit = QLineEdit()
        self._expiry_edit.setPlaceholderText("YYYY-MM-DD")
        form.addRow("Expiry Date:", self._expiry_edit)
        self._on_license_type_changed(self._license_type_combo.currentText())

        self._wildcard_checkbox = QCheckBox(
            "⚠️ Any machine (wildcard) — DEV/TEST ONLY, will be REJECTED by production builds"
        )
        form.addRow("", self._wildcard_checkbox)

        generate_button = QPushButton("🔏  Generate License Key")
        generate_button.setObjectName("GenerateButton")
        generate_button.clicked.connect(self._on_generate_license)
        form.addRow("", generate_button)

        return box

    def _build_output_section(self) -> QGroupBox:
        box = QGroupBox("3. Generated License Key")
        layout = QVBoxLayout(box)

        self._output_edit = QPlainTextEdit()
        self._output_edit.setObjectName("OutputEdit")
        self._output_edit.setReadOnly(True)
        self._output_edit.setPlaceholderText("Generated license key will appear here...")
        layout.addWidget(self._output_edit, stretch=1)

        copy_button = QPushButton("📋  Copy to Clipboard")
        copy_button.clicked.connect(self._on_copy_output)
        layout.addWidget(copy_button)

        layout.addWidget(QLabel("Session Log:"))
        self._log_edit = QTextEdit()
        self._log_edit.setObjectName("LogEdit")
        self._log_edit.setReadOnly(True)
        self._log_edit.setFixedHeight(140)
        layout.addWidget(self._log_edit)

        return box

    # ── Keypair Handling ─────────────────────────────────────────────────

    def _on_load_private_key(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Load Private Key", "", "PEM Files (*.pem);;All Files (*)"
        )
        if not path_str:
            return

        path = Path(path_str)
        try:
            pem_bytes = path.read_bytes()
            # Validate it actually loads as a private key before accepting it.
            load_pem_private_key(pem_bytes, password=None)
            self._private_key_pem = pem_bytes
            self._private_key_path = path
            self._keypair_status_label.setText(f"✅ Loaded: {path.name}")
            self._append_log(f"Loaded private key from {path}")
        except Exception as e:
            QMessageBox.critical(self, "Invalid Key File", f"Could not load private key:\n{e}")

    def _on_generate_keypair(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Generate New Keypair",
            "This creates a brand-new signing keypair.\n\n"
            "You will need to embed the new PUBLIC key into "
            "src/licensing/validator.py (EMBEDDED_PUBLIC_KEY_PEM) and "
            "rebuild the customer application, or existing customer "
            "installs won't recognize keys signed with it.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        private_key = ec.generate_private_key(ec.SECP256R1())
        private_pem = private_key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
        )
        public_pem = private_key.public_key().public_bytes(
            Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
        )

        save_path_str, _ = QFileDialog.getSaveFileName(
            self, "Save Private Key", "tilevision_private_key.pem", "PEM Files (*.pem)"
        )
        if save_path_str:
            Path(save_path_str).write_bytes(private_pem)
            self._private_key_path = Path(save_path_str)

        self._private_key_pem = private_pem
        self._keypair_status_label.setText(
            f"✅ New keypair generated{' — saved to ' + save_path_str if save_path_str else ' (not saved to disk)'}"
        )

        self._output_edit.setPlainText(
            "── PASTE THIS INTO src/licensing/validator.py — EMBEDDED_PUBLIC_KEY_PEM ──\n\n"
            + public_pem.decode("utf-8")
        )
        self._append_log("Generated a new signing keypair. Public key shown in output panel.")

    # ── License Generation ───────────────────────────────────────────────

    def _on_license_type_changed(self, license_type: str) -> None:
        self._expiry_edit.setText(compute_expiry_date(license_type))

    def _on_generate_license(self) -> None:
        if not self._private_key_pem:
            QMessageBox.warning(self, "No Signing Key", "Load or generate a private key first.")
            return

        customer_name = self._customer_name_edit.text().strip()
        machine_id = self._machine_id_edit.text().strip()
        license_type = self._license_type_combo.currentText()
        expires_at = self._expiry_edit.text().strip()
        use_wildcard = self._wildcard_checkbox.isChecked()

        if not customer_name:
            QMessageBox.warning(self, "Missing Field", "Customer Name is required.")
            return
        if not use_wildcard and not machine_id:
            QMessageBox.warning(
                self, "Missing Field",
                "Machine ID is required (or check the wildcard box for dev/test only)."
            )
            return
        try:
            datetime.strptime(expires_at, "%Y-%m-%d")
        except ValueError:
            QMessageBox.warning(self, "Invalid Date", "Expiry date must be in YYYY-MM-DD format.")
            return

        hardware_hash = "*" if use_wildcard else machine_id

        try:
            key = generate_license_key(
                self._private_key_pem, customer_name, expires_at, hardware_hash, license_type
            )
        except Exception as e:
            QMessageBox.critical(self, "Generation Failed", f"Could not generate license key:\n{e}")
            return

        self._output_edit.setPlainText(key)
        self._append_log(
            f"Generated {license_type} key for '{customer_name}' "
            f"(machine: {'ANY (wildcard)' if use_wildcard else machine_id[:16] + '...'}, "
            f"expires: {expires_at})"
        )

    def _on_copy_output(self) -> None:
        text = self._output_edit.toPlainText().strip()
        if not text:
            return
        clipboard: QClipboard = QGuiApplication.clipboard()
        clipboard.setText(text)
        self._append_log("Copied output to clipboard.")

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log_edit.append(f"[{timestamp}] {message}")

    # ── Styling ──────────────────────────────────────────────────────────

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background-color: #1A1D26; }
            QWidget { color: #E8EAF6; font-size: 12px; }
            #Title { font-size: 18px; font-weight: 700; }
            #Warning { color: #FFB74D; font-size: 11px; padding: 6px; background-color: #2A2418; border-radius: 6px; }
            QGroupBox {
                border: 1px solid #2D3250; border-radius: 8px; margin-top: 12px; padding-top: 12px;
                font-weight: 600;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #7C83D3; }
            QLineEdit, QComboBox, QPlainTextEdit, QTextEdit {
                background-color: #252837; border: 1px solid #3D4166; border-radius: 6px;
                padding: 6px; color: #E8EAF6;
            }
            #OutputEdit { font-family: "Consolas", monospace; font-size: 11px; }
            #LogEdit { font-family: "Consolas", monospace; font-size: 10px; color: #9E9E9E; }
            QPushButton {
                background-color: #2D3250; border: 1px solid #3D4166; border-radius: 6px;
                padding: 8px 14px; color: #E8EAF6;
            }
            QPushButton:hover { background-color: #3D4166; }
            #GenerateButton { background: #3949AB; font-weight: bold; }
            #GenerateButton:hover { background: #5C6BC0; }
            """
        )


def main() -> int:
    app = QApplication(sys.argv)
    window = AdminLicenseWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
