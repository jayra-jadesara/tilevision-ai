"""
Help / User Guide View for TileVision AI (Task E: Help / User Guide).

A modal dialog walking a new user through the core workflow in 5 steps:
choose a folder, index images, upload a customer photo, view similar
tiles, and double-click to open the match. Each step has a placeholder
for a screenshot — actual screenshots should be captured from a real
installation and dropped into src/resources/help/ once available; this
view references them by a conventional filename per step so wiring real
images in later is a one-line change (see _screenshot_path()).
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

logger = logging.getLogger("tilevision.presentation.views.help_view")


class _HelpStep(NamedTuple):
    number: int
    title: str
    description: str
    screenshot_filename: str


_STEPS: List[_HelpStep] = [
    _HelpStep(
        1, "Choose a Folder",
        "Go to the Index page and click Browse to select the folder containing "
        "your tile photos. TileVision AI scans it recursively, so subfolders "
        "are included automatically.",
        "step1_choose_folder.png",
    ),
    _HelpStep(
        2, "Index Images",
        "Click Start Indexing. Each image is analyzed and added to the search "
        "index in the background — you'll see live progress, and can pause, "
        "resume, or cancel at any time. Re-running it later only processes "
        "new or changed files.",
        "step2_index_images.png",
    ),
    _HelpStep(
        3, "Upload a Customer Image",
        "Go to the Search page and drag in a photo — from a customer's phone, "
        "a WhatsApp message, or your own camera — or click Browse to pick a "
        "file. You can also crop to just the tile pattern if the photo shows "
        "a whole room.",
        "step3_upload_customer_image.png",
    ),
    _HelpStep(
        4, "View Similar Tiles",
        "Matching tiles from your catalog appear ranked by similarity, with "
        "the closest match highlighted as \"Best Match.\" Filter by brand, "
        "category, color, or size to narrow the results.",
        "step4_view_similar_tiles.png",
    ),
    _HelpStep(
        5, "Double-Click to Open",
        "Double-click any result to open the full image in your default "
        "viewer. Right-click for more options: open the containing folder, "
        "or copy the file path.",
        "step5_double_click_to_open.png",
    ),
]


class HelpView(QDialog):
    """Modal Help / User Guide dialog."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Help & User Guide")
        self.resize(720, 640)
        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        title = QLabel("📖  Getting Started with TileVision AI")
        title.setObjectName("Title")
        layout.addWidget(title)

        subtitle = QLabel("Five steps from a folder of tile photos to instant visual search.")
        subtitle.setObjectName("Subtitle")
        layout.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        steps_container = QWidget()
        steps_layout = QVBoxLayout(steps_container)
        steps_layout.setSpacing(14)

        for step in _STEPS:
            steps_layout.addWidget(self._build_step_widget(step))
        steps_layout.addStretch()

        scroll.setWidget(steps_container)
        layout.addWidget(scroll, stretch=1)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_button = QPushButton("Close")
        close_button.setObjectName("CloseButton")
        close_button.clicked.connect(self.accept)
        close_row.addWidget(close_button)
        layout.addLayout(close_row)

    def _build_step_widget(self, step: _HelpStep) -> QWidget:
        card = QFrame()
        card.setObjectName("StepCard")
        layout = QHBoxLayout(card)
        layout.setSpacing(14)

        badge = QLabel(str(step.number))
        badge.setObjectName("StepBadge")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(36, 36)
        layout.addWidget(badge, alignment=Qt.AlignmentFlag.AlignTop)

        text_col = QVBoxLayout()
        title_label = QLabel(step.title)
        title_label.setObjectName("StepTitle")
        text_col.addWidget(title_label)

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
                scaled = pixmap.scaledToWidth(560, Qt.TransformationMode.SmoothTransformation)
                label.setPixmap(scaled)
                return label

        placeholder = QLabel(f"🖼️  Screenshot placeholder — {step.screenshot_filename}")
        placeholder.setObjectName("ScreenshotPlaceholder")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setFixedHeight(120)
        return placeholder

    @staticmethod
    def _screenshot_path(filename: str) -> Path:
        return Path(__file__).resolve().parents[2] / "resources" / "help" / filename

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QDialog { background-color: #1A1D26; }
            QWidget { color: #E8EAF6; }
            #Title { font-size: 18px; font-weight: 700; }
            #Subtitle { color: #8A8FA3; font-size: 12px; }
            #StepCard {
                background-color: #232634; border: 1px solid #2E3243; border-radius: 10px;
                padding: 14px;
            }
            #StepBadge {
                background-color: #3949AB; color: white; border-radius: 18px;
                font-weight: 700; font-size: 14px;
            }
            #StepTitle { font-size: 14px; font-weight: 700; color: #E8EAF6; }
            #StepDescription { font-size: 12px; color: #C7CAD9; padding-top: 2px; }
            #ScreenshotPlaceholder {
                background-color: #14161F; border: 1px dashed #3A3F52; border-radius: 6px;
                color: #55596B; font-size: 11px; margin-top: 8px;
            }
            #CloseButton {
                background-color: #2D3250; border-radius: 6px; padding: 8px 20px;
                font-weight: 600;
            }
            #CloseButton:hover { background-color: #3D4166; }
            """
        )
