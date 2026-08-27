"""
Help / User Guide page for TileVision AI.

A clear, full-width walkthrough: numbered steps, short actions, and sharp
content-only screenshots (no nested sidebar).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, NamedTuple, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QWidget,
    QFrame,
    QSizePolicy,
)

from src.theme.theme_manager import get_palette, get_shared_view_qss

logger = logging.getLogger("tilevision.presentation.views.help_view")

_RESOURCES_DIR = Path(__file__).resolve().parents[2] / "resources"
_LOGO_SMALL_PATH = _RESOURCES_DIR / "logo_small.png"

_COMPANY_NAME = "JD Software"
_CONTACT_NUMBER = "88662 77767"

# Display screenshots near full content width; render at width * max(dpr, 2).
_SCREENSHOT_MAX_LOGICAL_WIDTH = 980


class _HelpStep(NamedTuple):
    number: int
    title: str
    action: str  # one clear action line
    tip: str  # optional supporting tip
    screenshot_filename: str


_STEPS: List[_HelpStep] = [
    _HelpStep(
        1,
        "Pick your tile photos folder",
        "Go to Index → click Browse → choose the folder with your tile photos.",
        "You can add more folders later from Settings.",
        "step1_choose_folder.png",
    ),
    _HelpStep(
        2,
        "Start indexing",
        "Click Start Indexing. TileVision AI learns every tile in that folder.",
        "Watch progress here. You can Pause or Cancel anytime.",
        "step2_index_images.png",
    ),
    _HelpStep(
        3,
        "Drop in a customer photo",
        "Go to Search → drag a photo in, or click Browse to pick one.",
        "WhatsApp images, phone photos, and crops all work.",
        "step3_upload_customer_image.png",
    ),
    _HelpStep(
        4,
        "See matching tiles",
        "Closest matches appear first. Use Brand / Category / Color / Size filters if needed.",
        "Similarity % shows how close each tile is to the photo.",
        "step4_view_similar_tiles.png",
    ),
    _HelpStep(
        5,
        "Open the match",
        "Double-click a result to open the full photo.",
        "Right-click for Open Folder or Copy Path.",
        "step5_double_click_to_open.png",
    ),
]


class HelpView(QWidget):
    """Help / User Guide content page (stack page, not a dialog)."""

    def __init__(self, parent: Optional[QWidget] = None, theme: str = "dark") -> None:
        super().__init__(parent)
        self._theme = theme
        self.setObjectName("HelpView")
        self._setup_ui()
        self._apply_styles()

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        self._apply_styles()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 14)
        layout.setSpacing(12)

        layout.addWidget(self._build_header())

        scroll = QScrollArea()
        scroll.setObjectName("HelpScroll")
        scroll.viewport().setObjectName("HelpViewport")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        steps_container = QWidget()
        steps_container.setObjectName("HelpContent")
        steps_layout = QVBoxLayout(steps_container)
        steps_layout.setContentsMargins(4, 4, 4, 16)
        steps_layout.setSpacing(18)

        for step in _STEPS:
            steps_layout.addWidget(self._build_step_widget(step))
        steps_layout.addStretch(1)

        scroll.setWidget(steps_container)
        layout.addWidget(scroll, stretch=1)
        layout.addWidget(self._build_footer())

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("HelpHeader")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(4)

        title = QLabel("How TileVision AI Works")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "Follow these 5 steps — from your tile folder to finding a match."
        )
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(subtitle)

        return header

    def _build_step_widget(self, step: _HelpStep) -> QWidget:
        card = QFrame()
        card.setObjectName("StepCard")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        # Top row: number + titles
        top = QHBoxLayout()
        top.setSpacing(14)

        badge = QLabel(str(step.number))
        badge.setObjectName("StepBadge")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(40, 40)
        top.addWidget(badge, alignment=Qt.AlignmentFlag.AlignTop)

        text_col = QVBoxLayout()
        text_col.setSpacing(4)

        title_label = QLabel(step.title)
        title_label.setObjectName("StepTitle")
        title_label.setWordWrap(True)
        text_col.addWidget(title_label)

        action_label = QLabel(step.action)
        action_label.setObjectName("StepAction")
        action_label.setWordWrap(True)
        text_col.addWidget(action_label)

        tip_label = QLabel(step.tip)
        tip_label.setObjectName("StepTip")
        tip_label.setWordWrap(True)
        text_col.addWidget(tip_label)

        top.addLayout(text_col, stretch=1)
        layout.addLayout(top)

        shot = self._build_screenshot(step)
        layout.addWidget(shot)

        return card

    def _screenshot_display_width(self) -> int:
        # Prefer a wide, readable sheet; fall back when the page is still narrow.
        page_w = self.width() if self.width() > 200 else 1100
        available = max(860, page_w - 100)
        return min(_SCREENSHOT_MAX_LOGICAL_WIDTH, available)

    def _build_screenshot(self, step: _HelpStep) -> QWidget:
        path = self._screenshot_path(step.screenshot_filename)
        frame = QFrame()
        frame.setObjectName("ScreenshotFrame")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)

        if path.exists():
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                label = QLabel()
                label.setObjectName("ScreenshotImage")
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setScaledContents(False)

                # Always render ≥2x pixels for crisp text on Windows (often DPR≈1).
                dpr = max(2.0, float(self.devicePixelRatioF() or 1.0))
                logical_w = self._screenshot_display_width()
                target_px = int(logical_w * dpr)
                scaled = pixmap.scaledToWidth(
                    target_px, Qt.TransformationMode.SmoothTransformation
                )
                scaled.setDevicePixelRatio(dpr)
                label.setPixmap(scaled)
                label.setMinimumWidth(min(logical_w, scaled.width() / dpr))
                frame_layout.addWidget(label)
                return frame

        placeholder = QLabel("Screenshot coming soon")
        placeholder.setObjectName("ScreenshotPlaceholder")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setFixedHeight(80)
        frame_layout.addWidget(placeholder)
        return frame

    @staticmethod
    def _screenshot_path(filename: str) -> Path:
        return _RESOURCES_DIR / "help" / filename

    def _build_footer(self) -> QWidget:
        footer = QFrame()
        footer.setObjectName("Footer")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(2, 10, 2, 2)
        layout.setSpacing(10)

        logo_label = QLabel()
        pixmap = QPixmap(str(_LOGO_SMALL_PATH))
        if not pixmap.isNull():
            dpr = max(1.0, float(self.devicePixelRatioF() or 1.0))
            size_px = int(40 * dpr)
            scaled = pixmap.scaled(
                size_px,
                size_px,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            scaled.setDevicePixelRatio(dpr)
            logo_label.setPixmap(scaled)
            layout.addWidget(logo_label)

        credit_label = QLabel(
            f"TileVision AI — made by {_COMPANY_NAME}  •  {_CONTACT_NUMBER}"
        )
        credit_label.setObjectName("CreditLabel")
        layout.addWidget(credit_label)
        layout.addStretch()
        return footer

    def _apply_styles(self) -> None:
        p = get_palette(self._theme)
        self.setStyleSheet(
            get_shared_view_qss(self._theme)
            + f"""
            #HelpView {{ background-color: {p['bg_app']}; }}
            #HelpContent {{ background-color: {p['bg_app']}; }}
            #HelpViewport {{ background-color: {p['bg_app']}; }}
            #PageTitle {{
                font-size: 24px; font-weight: 750; color: {p['text_primary']};
            }}
            #PageSubtitle {{
                font-size: 14px; color: {p['text_secondary']}; padding-bottom: 4px;
            }}
            #StepCard {{
                background-color: {p['bg_panel']};
                border: 1px solid {p['border']};
                border-radius: 14px;
            }}
            #StepBadge {{
                background-color: {p['accent']};
                color: {p['button_text']};
                border-radius: 20px;
                font-size: 16px;
                font-weight: 800;
            }}
            #StepTitle {{
                font-size: 17px; font-weight: 750; color: {p['text_primary']};
            }}
            #StepAction {{
                font-size: 14px; font-weight: 600; color: {p['text_primary']};
                padding-top: 2px;
            }}
            #StepTip {{
                font-size: 13px; color: {p['text_secondary']};
            }}
            #ScreenshotFrame {{
                background-color: {p['bg_panel_alt']};
                border: 1px solid {p['border_strong']};
                border-radius: 10px;
                padding: 8px;
            }}
            #ScreenshotPlaceholder {{
                background-color: {p['bg_panel_alt']};
                border: 1px dashed {p['border_strong']};
                border-radius: 8px;
                color: {p['text_faint']};
                font-size: 12px;
            }}
            #Footer {{ border-top: 1px solid {p['border']}; margin-top: 2px; }}
            #CreditLabel {{ color: {p['text_muted']}; font-size: 12px; }}
            """
        )
