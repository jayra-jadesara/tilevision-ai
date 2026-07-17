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

        section_title = QLabel("Step 1 — Your Machine ID")
        section_title.setObjectName("SectionTitle")
        section_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))

        info_text = QLabel(
            "Copy your Machine ID and send it to your TileVision vendor. "
            "They will generate a license key for this PC."
        )
        info_text.setObjectName("InfoText")
        info_text.setWordWrap(True)

        hw_row = QHBoxLayout()
        hw_row.setSpacing(8)

        self._hw_id_edit = QLineEdit()
        self._hw_id_edit.setObjectName("HwIdEdit")
        self._hw_id_edit.setReadOnly(True)
        self._hw_id_edit.setPlaceholderText("Computing hardware fingerprint...")

        copy_button = QPushButton("Copy Machine ID")
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

        section_title = QLabel("Step 2 — Enter License Key")
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
            self._show_status("Machine ID copied. Send it to your vendor.", error=False)
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


class LicenseStartupChoiceDialog(QDialog):
    """
    First-run dialog: user chooses 15-day trial or license activation.

    The trial is not started automatically — the caller must invoke
    start_trial_access() only after the user picks the trial option.
    """

    def __init__(
        self,
        validate_use_case: ValidateLicenseUseCase,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._use_case = validate_use_case
        self._choice: Optional[str] = None

        self.setWindowTitle("TileVision AI — Get Started")
        self.setFixedSize(540, 460)
        self.setModal(True)
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.WindowCloseButtonHint |
            Qt.WindowType.MSWindowsFixedSizeDialogHint
        )

        self._setup_ui()
        self._apply_styles()
        self._load_hardware_id()

    @property
    def choice(self) -> Optional[str]:
        """'trial', 'license', or None if the dialog was dismissed."""
        return self._choice

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(16)

        title = QLabel("Welcome to TileVision AI")
        title.setObjectName("DialogTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        subtitle = QLabel(
            "How would you like to get started?\n"
            "Choose a free trial or activate with your license key."
        )
        subtitle.setObjectName("DialogSubtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        trial_btn = QPushButton("Start 15-Day Free Trial")
        trial_btn.setObjectName("TrialButton")
        trial_btn.setFixedHeight(52)
        trial_btn.clicked.connect(self._on_trial_clicked)
        layout.addWidget(trial_btn)

        trial_hint = QLabel("No license key needed — full access for 15 days on this PC.")
        trial_hint.setObjectName("ChoiceHint")
        trial_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        trial_hint.setWordWrap(True)
        layout.addWidget(trial_hint)

        layout.addSpacing(12)

        license_btn = QPushButton("I Have a License Key")
        license_btn.setObjectName("LicenseButton")
        license_btn.setFixedHeight(52)
        license_btn.clicked.connect(self._on_license_clicked)
        layout.addWidget(license_btn)

        license_hint = QLabel("Enter the key your TileVision vendor sent you.")
        license_hint.setObjectName("ChoiceHint")
        license_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        license_hint.setWordWrap(True)
        layout.addWidget(license_hint)

        layout.addStretch()

        machine_frame = QFrame()
        machine_frame.setObjectName("InfoFrame")
        machine_layout = QVBoxLayout(machine_frame)
        machine_layout.setContentsMargins(12, 10, 12, 10)
        machine_layout.setSpacing(6)

        machine_title = QLabel("Your Machine ID (for license requests)")
        machine_title.setObjectName("SectionTitle")
        machine_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))

        machine_row = QHBoxLayout()
        machine_row.setSpacing(8)
        self._hw_id_edit = QLineEdit()
        self._hw_id_edit.setObjectName("HwIdEdit")
        self._hw_id_edit.setReadOnly(True)
        copy_btn = QPushButton("Copy")
        copy_btn.setObjectName("CopyButton")
        copy_btn.setFixedWidth(72)
        copy_btn.clicked.connect(self._copy_hw_id)
        machine_row.addWidget(self._hw_id_edit)
        machine_row.addWidget(copy_btn)

        machine_layout.addWidget(machine_title)
        machine_layout.addLayout(machine_row)
        layout.addWidget(machine_frame)

    def _load_hardware_id(self) -> None:
        try:
            self._hw_id_edit.setText(self._use_case.get_hardware_fingerprint())
        except Exception:
            self._hw_id_edit.setText("")

    def _copy_hw_id(self) -> None:
        machine_id = self._hw_id_edit.text().strip()
        if not machine_id:
            QMessageBox.warning(self, "Unavailable", "Machine ID could not be read on this PC.")
            return
        QGuiApplication.clipboard().setText(machine_id)
        QMessageBox.information(
            self,
            "Copied",
            "Machine ID copied. Send it to your vendor to receive a license key.",
        )

    @Slot()
    def _on_trial_clicked(self) -> None:
        self._choice = "trial"
        self.accept()

    @Slot()
    def _on_license_clicked(self) -> None:
        self._choice = "license"
        self.accept()

    def _apply_styles(self) -> None:
        self.setStyleSheet("""
            QDialog {
                background-color: #1A1D26;
                color: #E8EAF6;
            }
            #DialogTitle {
                color: #E8EAF6;
            }
            #DialogSubtitle {
                color: #9E9E9E;
                font-size: 12px;
            }
            #ChoiceHint {
                color: #7DD3FC;
                font-size: 11px;
            }
            #InfoFrame {
                background-color: #1E2130;
                border: 1px solid #2D3250;
                border-radius: 8px;
            }
            #SectionTitle {
                color: #38BDF8;
                font-size: 11px;
            }
            #HwIdEdit {
                background-color: #252837;
                border: 1px solid #3D4166;
                border-radius: 6px;
                color: #B0BEC5;
                font-family: "Consolas", "Courier New", monospace;
                font-size: 11px;
                padding: 6px 10px;
            }
            #CopyButton {
                background-color: #2D3250;
                border: 1px solid #3D4166;
                border-radius: 6px;
                color: #B0BEC5;
            }
            #CopyButton:hover {
                background-color: #3D4166;
                color: #E8EAF6;
            }
            #TrialButton {
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
            #TrialButton:hover {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #0EA5E9,
                    stop: 1 #38BDF8
                );
            }
            #LicenseButton {
                background-color: #252837;
                border: 2px solid #3D4166;
                border-radius: 10px;
                color: #E8EAF6;
                font-size: 15px;
                font-weight: bold;
            }
            #LicenseButton:hover {
                border-color: #0EA5E9;
                color: #FFFFFF;
            }
        """)


class MachineIdWelcomeDialog(QDialog):
    """Shown once on first trial start so customers know how to request a license."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("TileVision AI — Get Your License")
        self.setFixedSize(520, 320)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        title = QLabel("15-Day Trial Started")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        steps = QLabel(
            "To buy a full license later:\n"
            "1. Copy your Machine ID below\n"
            "2. Send it to your TileVision vendor\n"
            "3. Paste the license key in Settings or restart the app"
        )
        steps.setWordWrap(True)
        layout.addWidget(steps)

        self._machine_id_edit = QLineEdit()
        self._machine_id_edit.setReadOnly(True)
        self._machine_id_edit.setPlaceholderText("Loading Machine ID...")
        layout.addWidget(self._machine_id_edit)

        copy_btn = QPushButton("Copy Machine ID")
        copy_btn.clicked.connect(self._copy_machine_id)
        layout.addWidget(copy_btn)

        layout.addStretch()
        ok_btn = QPushButton("Continue with Trial")
        ok_btn.setMinimumHeight(40)
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn)

        self._load_machine_id()

    def _load_machine_id(self) -> None:
        try:
            from src.licensing.hardware import get_machine_fingerprint
            self._machine_id_edit.setText(get_machine_fingerprint())
        except Exception:
            self._machine_id_edit.setText("")

    def _copy_machine_id(self) -> None:
        machine_id = self._machine_id_edit.text().strip()
        if not machine_id:
            QMessageBox.warning(self, "Unavailable", "Machine ID could not be read on this PC.")
            return
        QGuiApplication.clipboard().setText(machine_id)
        QMessageBox.information(
            self,
            "Copied",
            "Machine ID copied. Send it to your vendor to receive a license key.",
        )
