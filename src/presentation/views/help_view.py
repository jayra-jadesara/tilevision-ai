"""
Help / User Guide page for TileVision AI.

Numbered walkthrough with one action, one tip, and a real-UI screenshot
per step. Light and dark screenshot sets follow the active theme.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import List, NamedTuple, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QWidget,
    QFrame,
    QSizePolicy,
    QPushButton,
)

from src.theme.theme_manager import get_palette, get_shared_view_qss

logger = logging.getLogger("tilevision.presentation.views.help_view")

_RESOURCES_DIR = Path(__file__).resolve().parents[2] / "resources"
_LOGO_SMALL_PATH = _RESOURCES_DIR / "logo_small.png"
_HELP_DIR = _RESOURCES_DIR / "help"

_COMPANY_NAME = "Adesara Tech"
_CONTACT_NUMBER = "88662 77767"

# Display screenshots near full content width; render at width * max(dpr, 2).
_SCREENSHOT_MAX_LOGICAL_WIDTH = 980


class _HelpStep(NamedTuple):
    number: int
    title: str
    action: str
    tip: str
    screenshot_stem: str  # without theme / extension
    # Button labels that must still exist in the live UI (drift guard).
    ui_labels: tuple[str, ...] = ()


_STEPS: List[_HelpStep] = [
    _HelpStep(
        1,
        "Add folder & Index",
        "Go to Index → Browse → choose the folder with your tile photos, then Start Indexing.",
        "You can add more watched folders later from Settings.",
        "step01_add_folder_index",
        ("Browse", "Start Indexing"),
    ),
    _HelpStep(
        2,
        "Update Search Index vs Full rebuild",
        "In Settings → Search Index, use Update Search Index for new/changed files. Use Full rebuild only when you need a clean re-embed.",
        "Full rebuild re-processes every image and can take a long time on large catalogues — prefer Update when you can.",
        "step02_update_vs_rebuild",
        ("Update Search Index", "Full rebuild"),
    ),
    _HelpStep(
        3,
        "Search by photo",
        "Go to Search → drag a customer photo in, or click Browse.",
        "Phone photos, WhatsApp images, and crops all work.",
        "step03_search_by_photo",
        ("Browse",),
    ),
    _HelpStep(
        4,
        "Search with several photos",
        "Turn on multi-photo, drop 2+ photos of the same tile, then search.",
        "TileVision AI combines the photos for a stronger match.",
        "step04_multi_photo",
        (),
    ),
    _HelpStep(
        5,
        "Crop (Auto / Precise / Manual)",
        "Use Auto Crop Search for a quick tile crop, Precise Crop for AI edges, or Manual Crop and Search to draw the region.",
        "Cropping away floor/walls usually improves match %.",
        "step05_crop_modes",
        ("Auto Crop Search", "Precise Crop"),
    ),
    _HelpStep(
        6,
        "Search by description",
        "Type a short description (for example “matte grey subway”) and search.",
        "First use downloads a one-time description-search model (~580 MB) — internet needed once.",
        "step06_description_search",
        (),
    ),
    _HelpStep(
        7,
        "Read the results",
        "Check match %, +N copies, and use Preview / Copy / Reveal. Filter by brand, size, or detected colour.",
        "Similarity % is how close each catalogue tile is to your photo or description.",
        "step07_read_results",
        ("Export Catalogue",),
    ),
    _HelpStep(
        8,
        "Export Catalogue + Export Profiles",
        "Click Export Catalogue, pick a company profile, and save the PDF. Manage profiles under Settings → Export Profiles.",
        "Profiles store your logo, contact lines, max results, and “Also at” copy listing.",
        "step08_export_catalogue",
        ("Export Catalogue",),
    ),
    _HelpStep(
        9,
        "Duplicates",
        "Open Duplicates from the sidebar to find near-identical tiles in your catalogue.",
        "Review groups before deleting or consolidating folders.",
        "step09_duplicates",
        ("Duplicates",),
    ),
    _HelpStep(
        10,
        "History & Activity",
        "Open History for past searches and Activity for indexing, export, licence, and navigation events.",
        "Export logs or Copy Diagnostics from Settings when support asks.",
        "step10_history_activity",
        (),
    ),
    _HelpStep(
        11,
        "Settings",
        "Set theme (light/dark), watched folders, updates, and export logs.",
        "Minimize to tray and auto-index options live here too.",
        "step11_settings",
        (),
    ),
    _HelpStep(
        12,
        "Licence & support",
        "Open Licence to activate or renew. Contact your supplier with the Activity log if something fails.",
        f"Support: {_COMPANY_NAME} · {_CONTACT_NUMBER}",
        "step12_licence_support",
        (),
    ),
]


def help_button_labels_named_in_text() -> list[str]:
    """Flatten UI labels referenced by Help steps (for drift tests)."""
    labels: list[str] = []
    for step in _STEPS:
        labels.extend(step.ui_labels)
    # Also scan action/tip text for common controls named in copy.
    return sorted(set(labels))


class HelpView(QWidget):
    """Help / User Guide content page (stack page, not a dialog)."""

    def __init__(self, parent: Optional[QWidget] = None, theme: str = "dark") -> None:
        super().__init__(parent)
        self._theme = theme
        self.setObjectName("HelpView")
        self._step_anchors: dict[int, QWidget] = {}
        self._screenshot_labels: list[tuple[_HelpStep, QLabel]] = []
        self._setup_ui()
        self._apply_styles()

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        self._reload_screenshots()
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
        self._scroll = scroll

        steps_container = QWidget()
        steps_container.setObjectName("HelpContent")
        steps_layout = QVBoxLayout(steps_container)
        steps_layout.setContentsMargins(4, 4, 4, 16)
        steps_layout.setSpacing(18)

        steps_layout.addWidget(self._build_contents())
        steps_layout.addWidget(self._build_troubleshooting())

        for step in _STEPS:
            widget = self._build_step_widget(step)
            self._step_anchors[step.number] = widget
            steps_layout.addWidget(widget)
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
            "Twelve short steps — from indexing your catalogue to export, "
            "duplicates, history, and licence support."
        )
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(subtitle)
        return header

    def _build_contents(self) -> QWidget:
        box = QFrame()
        box.setObjectName("HelpContents")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)
        heading = QLabel("Contents")
        heading.setObjectName("HelpContentsTitle")
        layout.addWidget(heading)
        for step in _STEPS:
            # QPushButton treats '&' as a mnemonic marker ("Add folder & Index"
            # rendered as "Add folder _Index"). Escape as '&&'.
            link = QPushButton(f"{step.number}. {step.title}".replace("&", "&&"))
            link.setObjectName("HelpContentsLink")
            link.setCursor(Qt.CursorShape.PointingHandCursor)
            link.setFlat(True)
            link.setAutoDefault(False)
            link.setDefault(False)
            link.clicked.connect(
                lambda _checked=False, n=step.number: self._scroll_to_step(n)
            )
            layout.addWidget(link)
        return box

    def _build_troubleshooting(self) -> QWidget:
        box = QFrame()
        box.setObjectName("HelpTrouble")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)
        title = QLabel("Troubleshooting")
        title.setObjectName("HelpTroubleTitle")
        layout.addWidget(title)
        body = QLabel(
            "• Slow first description search — the model downloads once (~580 MB); "
            "stay online until it finishes.\n"
            "• Offline — photo search still works; description search needs the "
            "model on disk.\n"
            "• Rebuild taking a long time — prefer Update Search Index; Full rebuild "
            "re-embeds every image.\n"
            "• Where are the logs? — Settings → Export logs (tilevision.log).\n"
            "• Paste-back for support — Settings → Copy Diagnostics."
        )
        body.setObjectName("HelpTroubleBody")
        body.setWordWrap(True)
        layout.addWidget(body)
        return box

    def _scroll_to_step(self, number: int) -> None:
        widget = self._step_anchors.get(number)
        if widget is None:
            return

        def _go() -> None:
            self._scroll.ensureWidgetVisible(widget, 0, 40)

        QTimer.singleShot(0, _go)

    def _build_step_widget(self, step: _HelpStep) -> QWidget:
        card = QFrame()
        card.setObjectName("StepCard")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

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

    def _theme_dirname(self) -> str:
        return "dark" if str(self._theme).lower().startswith("dark") else "light"

    def _screenshot_path(self, stem: str) -> Path:
        themed = _HELP_DIR / self._theme_dirname() / f"{stem}.png"
        if themed.is_file():
            return themed
        # Fall back to legacy flat layout during migration.
        legacy = _HELP_DIR / f"{stem}.png"
        if legacy.is_file():
            return legacy
        return themed

    def _screenshot_display_width(self) -> int:
        page_w = self.width() if self.width() > 200 else 1100
        available = max(860, page_w - 100)
        return min(_SCREENSHOT_MAX_LOGICAL_WIDTH, available)

    def _build_screenshot(self, step: _HelpStep) -> QWidget:
        frame = QFrame()
        frame.setObjectName("ScreenshotFrame")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel()
        label.setObjectName("ScreenshotImage")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setScaledContents(False)
        self._screenshot_labels.append((step, label))
        self._apply_pixmap(step, label)
        frame_layout.addWidget(label)
        return frame

    def _apply_pixmap(self, step: _HelpStep, label: QLabel) -> None:
        path = self._screenshot_path(step.screenshot_stem)
        if path.exists():
            pixmap = QPixmap(str(path))
            if not pixmap.isNull():
                dpr = max(2.0, float(self.devicePixelRatioF() or 1.0))
                logical_w = self._screenshot_display_width()
                target_px = int(logical_w * dpr)
                scaled = pixmap.scaledToWidth(
                    target_px, Qt.TransformationMode.SmoothTransformation
                )
                scaled.setDevicePixelRatio(dpr)
                label.setPixmap(scaled)
                label.setMinimumWidth(min(logical_w, scaled.width() / dpr))
                label.setText("")
                return
        label.clear()
        label.setText("Screenshot coming soon")
        label.setObjectName("ScreenshotPlaceholder")
        label.setFixedHeight(80)

    def _reload_screenshots(self) -> None:
        for step, label in self._screenshot_labels:
            label.setObjectName("ScreenshotImage")
            label.setFixedHeight(16777215)  # reset max default
            self._apply_pixmap(step, label)

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

        credit_block = QVBoxLayout()
        credit_block.setSpacing(2)
        credit_label = QLabel(
            f"TileVision AI — made by {_COMPANY_NAME}  •  {_CONTACT_NUMBER}"
        )
        credit_label.setObjectName("CreditLabel")
        copyright_label = QLabel(
            f"© {date.today().year} {_COMPANY_NAME}. All rights reserved."
        )
        copyright_label.setObjectName("CopyrightLabel")
        credit_block.addWidget(credit_label)
        credit_block.addWidget(copyright_label)
        layout.addLayout(credit_block)
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
            #HelpContents, #HelpTrouble {{
                background-color: {p['bg_panel']};
                border: 1px solid {p['border']};
                border-radius: 12px;
            }}
            #HelpContentsTitle, #HelpTroubleTitle {{
                font-size: 15px; font-weight: 750; color: {p['text_primary']};
            }}
            #HelpContentsLink {{
                text-align: left; color: {p['accent']};
                font-size: 13px; padding: 2px 0;
            }}
            #HelpTroubleBody {{
                font-size: 13px; color: {p['text_secondary']};
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
            #CopyrightLabel {{ color: {p['text_faint']}; font-size: 11px; }}
            """
        )
