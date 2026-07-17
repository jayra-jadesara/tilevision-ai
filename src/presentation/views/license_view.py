"""
License Activation View for TileVision AI.

Provides a modal QDialog for entering and validating offline license keys.
Displays hardware fingerprint and activation status.

Design Decision:
    On startup, the MainWindow shows this dialog if no valid license is found.
    The dialog is modal — it blocks the main window until a valid license is entered
    or the user explicitly closes the application.

    Trial and full licenses use the same screen and the same key field; the vendor
    chooses the license type when generating the key.

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
    QPlainTextEdit,
    QPushButton,
    QFrame,
    QWidget,
    QMessageBox,
)

from src.core.use_cases.validate_license import ValidateLicenseUseCase
from src.theme.theme_manager import get_palette, get_shared_view_qss
from src.utils.brand_assets import logo_pixmap

logger = logging.getLogger("tilevision.presentation.views.license_view")


class LicenseView(QDialog):
    """
    Modal license activation dialog.

    Allows the user to:
      - View their hardware fingerprint (to send to the vendor for key generation).
      - Enter and validate an offline license key (trial or full).
      - Copy the fingerprint to the clipboard.
    """

    def __init__(
        self,
        validate_use_case: ValidateLicenseUseCase,
        theme: str = "light",
        show_back: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Initialize the LicenseView dialog.

        Args:
            validate_use_case: Fully configured license validation use case.
            theme: UI theme ("light" or "dark") matching app settings.
            show_back: When True, show a Back button that closes without activating.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._use_case = validate_use_case
        self._theme = theme if theme in ("light", "dark") else "light"
        self._palette = get_palette(self._theme)
        self._show_back = show_back
        self._is_activated = False

        self.setWindowTitle("TileVision AI — License Activation")
        self.setFixedSize(560, 580)
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

    @property
    def is_activated(self) -> bool:
        """True if a valid license key was successfully entered and saved."""
        return self._is_activated

    def _setup_ui(self) -> None:
        """Build and arrange all widgets within the dialog."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(20)

        layout.addWidget(self._build_header())
        layout.addWidget(self._build_hardware_id_section())
        layout.addWidget(self._build_license_input_section())

        self._status_label = QLabel("")
        self._status_label.setObjectName("StatusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        layout.addStretch()

        button_row = QHBoxLayout()
        button_row.setSpacing(12)

        if self._show_back:
            self._back_button = QPushButton("Back")
            self._back_button.setObjectName("SecondaryButton")
            self._back_button.setFixedHeight(48)
            self._back_button.setMinimumWidth(100)
            self._back_button.clicked.connect(self.reject)
            button_row.addWidget(self._back_button)

        button_row.addStretch()

        self._activate_button = QPushButton("Activate License")
        self._activate_button.setObjectName("ActivateButton")
        self._activate_button.setFixedHeight(48)
        self._activate_button.setMinimumWidth(200)
        self._activate_button.clicked.connect(self._on_activate_clicked)
        button_row.addWidget(self._activate_button)

        layout.addLayout(button_row)

    def _build_header(self) -> QWidget:
        """Build the product logo and title header."""
        container = QFrame()
        container.setObjectName("HeaderFrame")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        logo_label = QLabel()
        logo_scaled = logo_pixmap(64)
        if not logo_scaled.isNull():
            logo_label.setPixmap(logo_scaled)
        else:
            logo_label.setText("TileVision")
            logo_font = QFont()
            logo_font.setPointSize(18)
            logo_font.setBold(True)
            logo_label.setFont(logo_font)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_label = QLabel("TileVision AI")
        title_label.setObjectName("DialogTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

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

        info_text = QLabel(
            "Copy your Machine ID and send it to your TileVision vendor. "
            "They will generate a trial or full license key for this PC."
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
        copy_button.setObjectName("SecondaryButton")
        copy_button.setFixedWidth(120)
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

        hint = QLabel(
            "Paste the trial or full license key your vendor sent you. "
            "Both key types use this same field."
        )
        hint.setObjectName("InfoText")
        hint.setWordWrap(True)

        self._license_key_edit = QPlainTextEdit()
        self._license_key_edit.setObjectName("LicenseKeyEdit")
        self._license_key_edit.setPlaceholderText(
            "Paste your license key here..."
        )
        self._license_key_edit.setFixedHeight(96)
        self._license_key_edit.setTabChangesFocus(True)

        layout.addWidget(section_title)
        layout.addWidget(hint)
        layout.addWidget(self._license_key_edit)
        return container

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
                "Warning: Could not read hardware fingerprint. Contact support.", error=True
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
        """Validate the entered license key and accept the dialog if valid."""
        license_key = self._normalize_license_key(self._license_key_edit.toPlainText())

        if not license_key:
            self._show_status("Warning: Please enter a license key before activating.", error=True)
            return

        logger.info("License activation attempt started.")
        self._activate_button.setEnabled(False)
        self._activate_button.setText("Validating...")

        try:
            is_valid, message = self._use_case.validate_and_save(license_key)
        except Exception as e:
            logger.error(f"License validation raised an exception: {e}")
            is_valid, message = False, (
                "Invalid license key. Please check the key and try again, or contact support."
            )

        self._activate_button.setEnabled(True)
        self._activate_button.setText("Activate License")

        if is_valid:
            logger.info("License successfully activated.")
            self._is_activated = True
            self._show_status("License activated successfully! Welcome to TileVision AI.", error=False)
            QMessageBox.information(
                self,
                "License Activated",
                "Your license has been successfully activated.\n\nTileVision AI is now unlocked.",
            )
            self.accept()
        else:
            logger.warning(f"License key validation failed: {message}")
            self._show_status(message, error=True)
            self._license_key_edit.selectAll()
            self._license_key_edit.setFocus()

    @staticmethod
    def _normalize_license_key(raw: str) -> str:
        """Strip whitespace and line breaks from pasted license keys."""
        return "".join(raw.split())

    def _show_status(self, message: str, error: bool = False) -> None:
        """Display a status message below the license input field."""
        color = self._palette["danger_text"] if error else self._palette["success_text"]
        self._status_label.setText(f'<span style="color:{color};">{message}</span>')

    def _apply_styles(self) -> None:
        """Apply theme-aware styles matching the rest of the application."""
        p = self._palette
        self.setStyleSheet(
            get_shared_view_qss(self._theme)
            + f"""
            QDialog {{
                background-color: {p['bg_app']};
                color: {p['text_primary']};
            }}
            #Separator {{
                border: none;
                border-top: 1px solid {p['border']};
            }}
            #InfoFrame {{
                background-color: {p['bg_panel']};
                border: 1px solid {p['border']};
                border-radius: 8px;
            }}
            #SectionTitle {{
                color: {p['accent_text']};
                font-size: 12px;
                font-weight: bold;
            }}
            #InfoText {{
                color: {p['text_muted']};
                font-size: 11px;
            }}
            #HwIdEdit, #LicenseKeyEdit {{
                background-color: {p['bg_input']};
                border: 1px solid {p['border_strong']};
                border-radius: 6px;
                color: {p['text_primary']};
                font-family: "Consolas", "Courier New", monospace;
                font-size: 12px;
                padding: 8px 10px;
            }}
            #LicenseKeyEdit:focus, #HwIdEdit:focus {{
                border-color: {p['accent']};
            }}
            #StatusLabel {{
                font-size: 12px;
                min-height: 20px;
            }}
            QLabel {{
                color: {p['text_primary']};
            }}
            """
        )
