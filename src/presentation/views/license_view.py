"""
License Activation View for TileVision AI.

Provides a modal QDialog for entering and validating offline license keys.
Displays hardware fingerprint and activation status.

Design Decision:
    On startup, the MainWindow shows this dialog if no valid license is found.
    The dialog is modal — it blocks the main window until a valid license is entered
    or the user explicitly closes the application.
    
    Dependency on the ValidateLicenseUseCase is injected via the constructor,
    keeping this dialog testable and decoupled from implementation details.
"""

import logging
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QClipboard, QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFrame,
    QWidget,
    QMessageBox,
    QSizePolicy,
)

from src.core.use_cases.validate_license import ValidateLicenseUseCase

logger = logging.getLogger("tilevision.presentation.views.license_view")


class LicenseView(QDialog):
    """
    Modal license activation dialog.

    Allows the user to:
      - View their hardware fingerprint (to send to the vendor for key generation).
      - Enter and validate an offline license key.
      - Copy the fingerprint to the clipboard.
    """

    def __init__(
        self,
        validate_use_case: ValidateLicenseUseCase,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Initialize the LicenseView dialog.

        Args:
            validate_use_case: Fully configured license validation use case.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._use_case = validate_use_case
        self._is_activated = False

        self.setWindowTitle("TileVision AI — License Activation")
        self.setFixedSize(560, 480)
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.WindowCloseButtonHint |
            Qt.WindowType.MSWindowsFixedSizeDialogHint
        )
        self.setModal(True)

        self._setup_ui()
        self._apply_styles()
        self._load_hardware_id()

        logger.debug("LicenseView initialized.")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_activated(self) -> bool:
        """True if a valid license key was successfully entered and saved."""
        return self._is_activated

    # ── UI Construction ───────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        """Build and arrange all widgets within the dialog."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(20)

        # ── Logo / Brand Header
        layout.addWidget(self._build_header())

        # ── Hardware ID Section
        layout.addWidget(self._build_hardware_id_section())

        # ── License Key Input
        layout.addWidget(self._build_license_input_section())

        # ── Status Label
        self._status_label = QLabel("")
        self._status_label.setObjectName("StatusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        layout.addStretch()

        # ── Activate Button
        self._activate_button = QPushButton("Activate License")
        self._activate_button.setObjectName("ActivateButton")
        self._activate_button.setFixedHeight(48)
        self._activate_button.clicked.connect(self._on_activate_clicked)
        layout.addWidget(self._activate_button)

    def _build_header(self) -> QWidget:
        """Build the product logo and title header."""
        container = QFrame()
        container.setObjectName("HeaderFrame")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        logo_label = QLabel("🟦")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_font = QFont()
        logo_font.setPointSize(36)
        logo_label.setFont(logo_font)

        title_label = QLabel("TileVision AI")
        title_label.setObjectName("DialogTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)

        subtitle_label = QLabel("Offline Visual Tile Search Platform")
        subtitle_label.setObjectName("DialogSubtitle")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setObjectName("Separator")

        layout.addWidget(logo_label)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addWidget(separator)
        return container

    def _build_hardware_id_section(self) -> QFrame:
        """Build the hardware fingerprint display panel."""
        container = QFrame()
        container.setObjectName("InfoFrame")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        section_title = QLabel("🖥  Your Hardware Fingerprint")
        section_title.setObjectName("SectionTitle")
        section_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))

        info_text = QLabel(
            "Share this fingerprint with TileVision support to generate your license key:"
        )
        info_text.setObjectName("InfoText")
        info_text.setWordWrap(True)

        hw_row = QHBoxLayout()
        hw_row.setSpacing(8)

        self._hw_id_edit = QLineEdit()
        self._hw_id_edit.setObjectName("HwIdEdit")
        self._hw_id_edit.setReadOnly(True)
        self._hw_id_edit.setPlaceholderText("Computing hardware fingerprint...")

        copy_button = QPushButton("Copy")
        copy_button.setObjectName("CopyButton")
        copy_button.setFixedWidth(90)
        copy_button.setFixedHeight(32)
        copy_button.setToolTip("Copy fingerprint to clipboard")
        copy_button.clicked.connect(self._copy_hw_id)

        hw_row.addWidget(self._hw_id_edit)
        hw_row.addWidget(copy_button)

        layout.addWidget(section_title)
        layout.addWidget(info_text)
        layout.addLayout(hw_row)
        return container

    def _build_license_input_section(self) -> QFrame:
        """Build the license key entry field."""
        container = QFrame()
        container.setObjectName("InfoFrame")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        section_title = QLabel("🔑  Enter Your License Key")
        section_title.setObjectName("SectionTitle")
        section_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))

        self._license_key_edit = QLineEdit()
        self._license_key_edit.setObjectName("LicenseKeyEdit")
        self._license_key_edit.setPlaceholderText(
            "Paste your license key here (e.g. TVAI-XXXX-XXXX-XXXX-XXXX)"
        )
        self._license_key_edit.setFixedHeight(40)
        # Allow Enter key to trigger activation
        self._license_key_edit.returnPressed.connect(self._on_activate_clicked)

        layout.addWidget(section_title)
        layout.addWidget(self._license_key_edit)
        return container

    # ── Logic ─────────────────────────────────────────────────────────────────

    def _load_hardware_id(self) -> None:
        """Load and display the hardware fingerprint from the validation use case."""
        try:
            hw_id = self._use_case.get_hardware_fingerprint()
            self._hw_id_edit.setText(hw_id)
            logger.debug(f"Hardware fingerprint loaded: {hw_id[:8]}...")
        except Exception as e:
            logger.error(f"Failed to load hardware fingerprint: {e}")
            self._hw_id_edit.setText("ERROR: Unable to read hardware ID")
            self._show_status(
                "Warning: Warning: Could not read hardware fingerprint. Contact support.", error=True
            )

    @Slot()
    def _copy_hw_id(self) -> None:
        """Copy the hardware fingerprint to the system clipboard."""
        hw_id = self._hw_id_edit.text()
        if hw_id and not hw_id.startswith("ERROR"):
            clipboard: QClipboard = QGuiApplication.clipboard()
            clipboard.setText(hw_id)
            self._show_status("Hardware fingerprint copied to clipboard.", error=False)
            logger.info("Hardware fingerprint copied to clipboard.")
        else:
            self._show_status("Warning: Nothing to copy — fingerprint is unavailable.", error=True)

    @Slot()
    def _on_activate_clicked(self) -> None:
        """
        Validate the entered license key against the hardware fingerprint.
        Accept the dialog if the key is valid.
        """
        license_key = self._license_key_edit.text().strip()

        if not license_key:
            self._show_status("Warning: Please enter a license key before activating.", error=True)
            return

        logger.info("License activation attempt started.")
        self._activate_button.setEnabled(False)
        self._activate_button.setText("Validating...")

        try:
            is_valid = self._use_case.validate_and_save(license_key)
        except Exception as e:
            logger.error(f"License validation raised an exception: {e}")
            is_valid = False

        self._activate_button.setEnabled(True)
        self._activate_button.setText("Activate License")

        if is_valid:
            logger.info("License successfully activated.")
            self._is_activated = True
            self._show_status("License activated successfully! Welcome to TileVision AI.", error=False)
            # Small delay feedback before closing
            QMessageBox.information(
                self,
                "License Activated",
                "🎉 Your license has been successfully activated.\n\nTileVision AI is now unlocked.",
            )
            self.accept()
        else:
            logger.warning("License key validation failed.")
            self._show_status(
                "Invalid license key. Please check the key and try again, or contact support.",
                error=True,
            )
            self._license_key_edit.selectAll()
            self._license_key_edit.setFocus()

    def _show_status(self, message: str, error: bool = False) -> None:
        """
        Display a status message below the license input field.

        Args:
            message: The status message to display.
            error: If True, styles the label as an error (red), else success (green).
        """
        color = "#FF6B6B" if error else "#69F0AE"
        self._status_label.setText(f'<span style="color:{color};">{message}</span>')

    # ── Styling ───────────────────────────────────────────────────────────────

    def _apply_styles(self) -> None:
        """Apply QSS dark theme to the dialog."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1A1D26;
                color: #E8EAF6;
            }

            #DialogTitle {
                color: #E8EAF6;
                font-size: 18px;
                font-weight: bold;
            }
            #DialogSubtitle {
                color: #7DD3FC;
                font-size: 12px;
            }
            #Separator {
                border: none;
                border-top: 1px solid #2D3250;
            }

            #InfoFrame {
                background-color: #1E2130;
                border: 1px solid #2D3250;
                border-radius: 8px;
            }
            #SectionTitle {
                color: #38BDF8;
                font-size: 12px;
                font-weight: bold;
            }
            #InfoText {
                color: #9E9E9E;
                font-size: 11px;
            }

            #HwIdEdit {
                background-color: #252837;
                border: 1px solid #3D4166;
                border-radius: 6px;
                color: #B0BEC5;
                font-family: "Consolas", "Courier New", monospace;
                font-size: 12px;
                padding: 6px 10px;
            }

            #LicenseKeyEdit {
                background-color: #252837;
                border: 1px solid #3D4166;
                border-radius: 6px;
                color: #E8EAF6;
                font-family: "Consolas", "Courier New", monospace;
                font-size: 13px;
                letter-spacing: 2px;
                padding: 6px 10px;
            }
            #LicenseKeyEdit:focus {
                border-color: #0EA5E9;
            }

            #CopyButton {
                background-color: #2D3250;
                border: 1px solid #3D4166;
                border-radius: 6px;
                color: #B0BEC5;
                font-size: 12px;
            }
            #CopyButton:hover {
                background-color: #3D4166;
                color: #E8EAF6;
            }

            #ActivateButton {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #0369A1,
                    stop: 1 #0EA5E9
                );
                border: none;
                border-radius: 10px;
                color: #FFFFFF;
                font-size: 15px;
                font-weight: bold;
            }
            #ActivateButton:hover {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #0EA5E9,
                    stop: 1 #38BDF8
                );
            }
            #ActivateButton:pressed {
                background-color: #283593;
            }
            #ActivateButton:disabled {
                background-color: #2D3250;
                color: #546E7A;
            }

            #StatusLabel {
                font-size: 12px;
                min-height: 20px;
            }

            QLabel {
                color: #E8EAF6;
            }
        """)
