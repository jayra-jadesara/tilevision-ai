"""
Help / User Guide View for TileVision AI (Task E: Help / User Guide).

A modal dialog walking a new user through the core workflow in 5 simple
steps: choose a folder, index images, upload a customer photo, view
similar tiles, and double-click to open the match.

Rewritten for clarity after user feedback that the original version was
"more difficult to understand" — plain, short sentences, one clear action
per step, and a visual step number instead of dense paragraphs. Ends with
a small branded footer (logo + "Made by JD Software" + contact number),
answering a second piece of feedback asking for that credit to appear
somewhere in the app.
"""

import logging
from pathlib import Path
from typing import List, NamedTuple, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QWidget,
    QFrame,
)

from src.theme.theme_manager import get_palette

logger = logging.getLogger("tilevision.presentation.views.help_view")

_RESOURCES_DIR = Path(__file__).resolve().parents[2] / "resources"
_LOGO_SMALL_PATH = _RESOURCES_DIR / "logo_small.png"

_COMPANY_NAME = "JD Software"
_CONTACT_NUMBER = "88662 77767"


class _HelpStep(NamedTuple):
    number: int
    icon: str
    title: str
    description: str
    screenshot_filename: str


# Kept deliberately short — one plain-language sentence per step, not a
# paragraph. If it doesn't fit on one line at a glance, it's too long.
_STEPS: List[_HelpStep] = [
    _HelpStep(
        1, "📂", "Pick Your Tile Photos Folder",
        "On the Index page, click Browse and choose the folder with your tile photos.",
        "step1_choose_folder.png",
    ),
    _HelpStep(
        2, "⚡", "Click Start Indexing",
        "TileVision AI reads every photo and learns what each tile looks like. "
        "You can watch the progress, or pause/cancel any time.",
        "step2_index_images.png",
    ),
    _HelpStep(
        3, "📸", "Drop In a Customer's Photo",
        "On the Search page, drag in a photo — from WhatsApp, a phone camera, "
        "anywhere — or click Browse to pick one.",
        "step3_upload_customer_image.png",
    ),
    _HelpStep(
        4, "✨", "See the Matching Tiles",
        "Your closest matches appear instantly, best match first. Use the filters "
        "to narrow by brand, color, or size.",
        "step4_view_similar_tiles.png",
    ),
    _HelpStep(
        5, "🖱️", "Double-Click to Open",
        "Double-click any result to open the full photo. Right-click for more — "
        "open its folder, or copy the file path.",
        "step5_double_click_to_open.png",
    ),
]


class HelpView(QDialog):
    """Modal Help / User Guide dialog."""

    def __init__(self, parent: Optional[QWidget] = None, theme: str = "dark") -> None:
        super().__init__(parent)
        self._theme = theme
        self.setWindowTitle("Help & User Guide")
        self.resize(900, 720)
        self.setMinimumSize(860, 680)
        self._setup_ui()
        self._apply_styles()

    def set_theme(self, theme: str) -> None:
        """Re-skin this dialog for a newly-selected theme."""
        self._theme = theme
        self._apply_styles()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 16)
        layout.setSpacing(10)

        layout.addWidget(self._build_header())

        scroll = QScrollArea()
        scroll.setObjectName("HelpScroll")
        scroll.viewport().setObjectName("HelpViewport")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setAutoFillBackground(False)

        steps_container = QWidget()
        steps_container.setObjectName("HelpContent")
        steps_container.setAutoFillBackground(True)
        steps_layout = QVBoxLayout(steps_container)
        steps_layout.setContentsMargins(10,10,10,10)
        steps_layout.setSpacing(10)

        for step in _STEPS:
            steps_layout.addWidget(self._build_step_widget(step))
        steps_layout.addStretch()

        scroll.setWidget(steps_container)
        layout.addWidget(scroll, stretch=1)

        layout.addWidget(self._build_footer())

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("HelpHeader")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(2)

        title = QLabel("How TileVision AI Works")
        title.setObjectName("Title")
        layout.addWidget(title)

        subtitle = QLabel("Five simple steps — from a folder of photos to instant matches.")
        subtitle.setObjectName("Subtitle")
        layout.addWidget(subtitle)

        return header

    def _build_step_widget(self, step: _HelpStep) -> QWidget:
        card = QFrame()
        card.setObjectName("StepCard")
        layout = QHBoxLayout(card)
        layout.setSpacing(14)

        badge = QLabel(step.icon)
        badge.setObjectName("StepBadge")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(44, 44)
        layout.addWidget(badge, alignment=Qt.AlignmentFlag.AlignTop)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)

        title_row = QHBoxLayout()
        step_number = QLabel(f"Step {step.number}")
        step_number.setObjectName("StepNumber")
        title_row.addWidget(step_number)
        title_label = QLabel(step.title)
        title_label.setObjectName("StepTitle")
        title_row.addWidget(title_label)
        title_row.addStretch()
        text_col.addLayout(title_row)

        desc_label = QLabel(step.description)
        desc_label.setObjectName("StepDescription")
        desc_label.setWordWrap(True)
        text_col.addWidget(desc_label)

        text_col.addWidget(self._build_screenshot_placeholder(step))
        layout.addLayout(text_col, stretch=1)

        return card

    def _build_screenshot_placeholder(self, step: _HelpStep) -> QWidget:
        """
        Shows the real screenshot if one has been dropped into
        src/resources/help/, otherwise a labeled placeholder box so the
        layout/spacing is correct ahead of screenshots being captured.
        """
        path = self._screenshot_path(step.screenshot_filename)
        if path.exists():
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                label = QLabel()
                scaled = pixmap.scaledToWidth(540, Qt.TransformationMode.SmoothTransformation)
                label.setPixmap(scaled)
                return label

        placeholder = QLabel("🖼️  Screenshot coming soon")
        placeholder.setObjectName("ScreenshotPlaceholder")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setFixedHeight(70)
        return placeholder

    @staticmethod
    def _screenshot_path(filename: str) -> Path:
        return _RESOURCES_DIR / "help" / filename

    def _build_footer(self) -> QWidget:
        footer = QFrame()
        footer.setObjectName("Footer")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(4, 10, 4, 4)
        layout.setSpacing(10)

        logo_label = QLabel()
        pixmap = QPixmap(str(_LOGO_SMALL_PATH))
        if not pixmap.isNull():
            logo_label.setPixmap(
                pixmap.scaled(48, 48, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
            layout.addWidget(logo_label)

        credit_label = QLabel(f"TileVision AI — made by {_COMPANY_NAME}  •  📞 {_CONTACT_NUMBER}")
        credit_label.setObjectName("CreditLabel")
        layout.addWidget(credit_label)

        layout.addStretch()

        close_button = QPushButton("Got it")
        close_button.setObjectName("CloseButton")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        return footer

    def _apply_styles(self) -> None:
        p = get_palette(self._theme)
        self.setStyleSheet(
            f"""
            QDialog {{ background-color: {p['bg_app']}; }}
            QWidget {{ color: {p['text_primary']}; }}
            #Title {{ font-size: 18px; font-weight: 700; }}
            #Subtitle {{ color: {p['text_muted']}; font-size: 12px; }}
            #StepCard {{
                background-color: {p['bg_panel']}; border: 1px solid {p['border']}; border-radius: 10px;
                padding: 12px;
            }}
            #StepBadge {{
                background-color: {p['accent']}; border-radius: 12px; font-size: 20px;
            }}
            #StepNumber {{
                color: {p['accent_text']}; font-size: 11px; font-weight: 700;
                text-transform: uppercase; margin-right: 6px;
            }}
            #StepTitle {{ font-size: 14px; font-weight: 700; color: {p['text_primary']}; }}
            #StepDescription {{ font-size: 12px; color: {p['text_secondary']}; padding-top: 2px; }}
            #ScreenshotPlaceholder {{
                background-color: {p['bg_panel_alt']}; border: 1px dashed {p['border_strong']}; border-radius: 6px;
                color: {p['text_faint']}; font-size: 11px; margin-top: 6px;
            }}
            #Footer {{ border-top: 1px solid {p['border']}; margin-top: 4px; }}
            #CreditLabel {{ color: {p['text_muted']}; font-size: 11px; }}
            #CloseButton {{
                background-color: {p['accent']}; border-radius: 6px; padding: 8px 20px;
                font-weight: 600; color: white;
            }}
            #CloseButton:hover {{ background-color: {p['accent_hover']}; }}
            """
        )
