"""
First-run setup wizard for TileVision AI.

Guides users through ordered dependency installation before the main
application loads. Skips steps that are already satisfied.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.config.settings import AppSettings
from src.theme.theme_manager import get_palette
from src.utils.brand_assets import logo_pixmap
from src.utils.platform_info import app_icon_path
from src.utils.dependency_check import (
    CURRENT_SETUP_VERSION,
    INSTALL_STEPS,
    StepStatus,
    all_dependencies_satisfied,
    check_all_steps,
    install_step_packages,
    step_is_complete,
)

logger = logging.getLogger("tilevision.presentation.views.setup_wizard")


class SetupWizardDialog(QDialog):
    """Modal first-run installer shown before the main window."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        theme: str = "light",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._theme = theme
        self._step_index = 0
        self._step_statuses: List[StepStatus] = check_all_steps()

        self.setWindowTitle("TileVision AI — First-Time Setup")
        self.setModal(True)
        self.setMinimumSize(640, 480)
        if (icon := app_icon_path()) is not None:
            self.setWindowIcon(QIcon(str(icon)))
        self._build_ui()
        self._apply_theme()
        self._refresh_step()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        header_row = QHBoxLayout()
        header_row.setSpacing(16)

        self._logo_label = QLabel()
        self._logo_label.setObjectName("WizardLogo")
        logo = logo_pixmap(72)
        if not logo.isNull():
            self._logo_label.setPixmap(logo)
            self._logo_label.setFixedSize(72, 72)
            self._logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            header_row.addWidget(self._logo_label)

        header_text = QVBoxLayout()
        header_text.setSpacing(4)
        self._headline = QLabel("Welcome to TileVision AI")
        self._headline.setObjectName("WizardTitle")
        header_text.addWidget(self._headline)

        self._subtitle = QLabel(
            "Ceramic tile visual search for showrooms and catalog teams. "
            "Complete each step below in order — downloads start only when needed."
        )
        self._subtitle.setObjectName("WizardSubtitle")
        self._subtitle.setWordWrap(True)
        header_text.addWidget(self._subtitle)
        header_row.addLayout(header_text, stretch=1)
        root.addLayout(header_row)

        self._step_title = QLabel("")
        self._step_title.setObjectName("WizardStepTitle")
        root.addWidget(self._step_title)

        self._step_description = QLabel("")
        self._step_description.setObjectName("WizardNote")
        self._step_description.setWordWrap(True)
        root.addWidget(self._step_description)

        self._package_list = QListWidget()
        self._package_list.setObjectName("WizardPackageList")
        root.addWidget(self._package_list, stretch=1)

        self._log_output = QTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setPlaceholderText("Install log will appear here...")
        self._log_output.setMaximumHeight(120)
        root.addWidget(self._log_output)

        button_row = QHBoxLayout()
        self._install_button = QPushButton("Install This Step")
        self._install_button.clicked.connect(self._on_install_clicked)
        button_row.addWidget(self._install_button)

        self._skip_button = QPushButton("Skip to Next Step")
        self._skip_button.clicked.connect(self._on_skip_clicked)
        button_row.addWidget(self._skip_button)

        button_row.addStretch()

        self._finish_button = QPushButton("Finish and Start TileVision")
        self._finish_button.clicked.connect(self._on_finish_clicked)
        self._finish_button.setEnabled(False)
        button_row.addWidget(self._finish_button)

        root.addLayout(button_row)

    def _apply_theme(self) -> None:
        palette = get_palette(self._theme)
        self.setStyleSheet(
            f"""
            QDialog {{ background-color: {palette['bg_app']}; color: {palette['text_primary']}; }}
            #WizardTitle {{ font-size: 20px; font-weight: 700; color: {palette['text_primary']}; }}
            #WizardSubtitle, #WizardNote {{ color: {palette['text_secondary']}; font-size: 12px; }}
            #WizardStepTitle {{ font-size: 15px; font-weight: 600; color: {palette['accent_text']}; }}
            QListWidget {{
                background-color: {palette['bg_panel']};
                border: 1px solid {palette['border']};
                border-radius: 6px;
                padding: 6px;
            }}
            QTextEdit {{
                background-color: {palette['bg_panel']};
                border: 1px solid {palette['border']};
                border-radius: 6px;
                font-family: Consolas, monospace;
                font-size: 11px;
            }}
            QPushButton {{
                background-color: {palette['button_bg']};
                color: {palette['button_secondary_text']};
                border: 1px solid {palette['border']};
                border-radius: 6px;
                padding: 8px 16px;
            }}
            QPushButton:hover {{ background-color: {palette['button_hover']}; }}
            QPushButton:disabled {{ color: {palette['text_faint']}; }}
            """
        )

    _STEP_SUMMARY_LABELS = {
        "core": "Application core",
        "ai_stack": "AI search engine",
        "gpu_optional": "GPU acceleration (optional)",
    }

    def _step_summary_label(self, step_id: str, fallback_title: str) -> str:
        return self._STEP_SUMMARY_LABELS.get(step_id, fallback_title)

    def _runtime_status_lines(self) -> list[tuple[str, bool]]:
        """Customer-visible runtime checks: Python and SQLite only."""
        if not self._step_statuses:
            return []
        runtime = self._step_statuses[0]
        lines: list[tuple[str, bool]] = []
        for pkg in runtime.packages:
            if pkg.installed:
                version = f" ({pkg.version})" if pkg.version else ""
                lines.append((f"{pkg.spec.display_name}{version} — OK", True))
            else:
                note = f" — {pkg.note}" if pkg.note else " — required"
                lines.append((f"{pkg.spec.display_name}{note}", False))
        return lines

    def _component_status_lines(self) -> list[tuple[str, bool]]:
        """High-level component rows — never list third-party library names."""
        lines: list[tuple[str, bool]] = []
        for status in self._step_statuses:
            if status.step.step_id == "runtime":
                continue
            label = self._step_summary_label(status.step.step_id, status.step.title)
            missing_required = [
                pkg
                for pkg in status.packages
                if not pkg.installed and not pkg.spec.optional and not pkg.spec.builtin
            ]
            if status.step.optional and not missing_required and not any(
                not pkg.installed for pkg in status.packages
            ):
                continue
            if status.step.optional and missing_required:
                lines.append((f"{label} — available to install", False))
            elif missing_required:
                lines.append((f"{label} — install required", False))
            elif step_is_complete(status):
                lines.append((f"{label} — Ready", True))
        return lines

    def _fill_status_list(self) -> None:
        self._package_list.clear()
        for text, ok in self._runtime_status_lines() + self._component_status_lines():
            item = QListWidgetItem(text)
            if ok:
                item.setForeground(Qt.GlobalColor.darkGreen)
            self._package_list.addItem(item)

    def _refresh_step(self) -> None:
        self._step_statuses = check_all_steps()
        while self._step_index < len(INSTALL_STEPS):
            if not step_is_complete(self._step_statuses[self._step_index]):
                break
            self._step_index += 1

        if self._step_index >= len(INSTALL_STEPS):
            self._show_complete_state()
            return

        status = self._step_statuses[self._step_index]
        step = status.step
        self._step_title.setText(step.title)
        self._step_description.setText(step.description)

        self._fill_status_list()

        complete = step_is_complete(status)
        is_builtin_only = all(pkg.spec.builtin for pkg in status.packages)
        self._install_button.setEnabled(not complete and not is_builtin_only)
        if status.step.optional:
            self._install_button.setText("Install GPU Acceleration")
        else:
            self._install_button.setText("Install This Step")
        self._skip_button.setEnabled(self._step_index < len(INSTALL_STEPS) - 1)
        self._finish_button.setEnabled(all_dependencies_satisfied())

    def _show_complete_state(self) -> None:
        self._step_title.setText("Setup Complete")
        self._step_description.setText(
            "All required components are installed. Click Finish to open TileVision AI."
        )
        self._fill_status_list()
        self._install_button.setEnabled(False)
        self._skip_button.setEnabled(False)
        self._finish_button.setEnabled(True)

    def _on_install_clicked(self) -> None:
        if self._step_index >= len(INSTALL_STEPS):
            return

        step = INSTALL_STEPS[self._step_index]
        self._install_button.setEnabled(False)
        self._log_output.append(f"Installing: {step.title}...")
        self.repaint()

        ok, message = install_step_packages(step)
        self._log_output.append(message)
        if not ok:
            QMessageBox.warning(
                self,
                "Install Failed",
                f"Could not install all packages for this step.\n\n{message}",
            )
        self._refresh_step()

    def _on_skip_clicked(self) -> None:
        if self._step_index < len(INSTALL_STEPS) - 1:
            self._step_index += 1
            self._refresh_step()

    def _on_finish_clicked(self) -> None:
        if not all_dependencies_satisfied():
            QMessageBox.warning(
                self,
                "Setup Incomplete",
                "Some required packages are still missing. Install all steps before continuing.",
            )
            return

        self._settings.setup_wizard_completed = True
        self._settings.setup_wizard_version = CURRENT_SETUP_VERSION
        logger.info("First-run setup wizard completed.")
        self.accept()


def should_show_setup_wizard(settings: AppSettings) -> bool:
    """Return True when the wizard must run before the main application."""
    import sys

    # Packaged builds bundle all dependencies — never run pip from the installer.
    if getattr(sys, "frozen", False):
        return False

    if not settings.setup_wizard_completed:
        return True
    if settings.setup_wizard_version < CURRENT_SETUP_VERSION:
        return not all_dependencies_satisfied()
    return not all_dependencies_satisfied()
