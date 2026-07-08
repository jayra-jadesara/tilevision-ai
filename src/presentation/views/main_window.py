"""
Main Application Window for TileVision AI.

Implements the primary QMainWindow containing the navigation sidebar
and a stacked content area for all major feature views.

Currently hosts:
    - Indexing View (Feature 1: Folder Indexing)

Future views (Feature 2+) will be added as tabs/navigation items.

Design Decision:
    Uses a left-side icon+text navigation panel (QFrame) rather than QTabWidget,
    providing a modern sidebar navigation UX consistent with commercial desktop apps.
    Each navigation item switches the central QStackedWidget to the appropriate view.
    All dependencies are constructor-injected, making the window fully testable.
"""

import logging
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QIcon, QAction, QCloseEvent
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QFrame,
    QLabel,
    QPushButton,
    QStackedWidget,
    QSizePolicy,
    QStatusBar,
    QMessageBox,
    QSpacerItem,
)

from src.presentation.views.indexing_view import IndexingView
from src.presentation.viewmodels.indexing_viewmodel import IndexingViewModel, IndexingState

logger = logging.getLogger("tilevision.presentation.views.main_window")


class NavButton(QPushButton):
    """
    A custom navigation sidebar button with icon and label.

    Provides consistent active/inactive styling for sidebar navigation.
    """

    def __init__(self, icon_text: str, label: str, parent: Optional[QWidget] = None) -> None:
        """
        Initialize a navigation button.

        Args:
            icon_text: Emoji or text icon displayed above the label.
            label: Button label text.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("NavButton")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_label = QLabel(icon_text)
        icon_label.setObjectName("NavIcon")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_font = QFont()
        icon_font.setPointSize(18)
        icon_label.setFont(icon_font)

        text_label = QLabel(label)
        text_label.setObjectName("NavLabel")
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(icon_label)
        layout.addWidget(text_label)

        # Prevent the internal labels from consuming mouse events (let button handle them)
        icon_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        text_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self.setFixedWidth(100)
        self.setMinimumHeight(80)


class MainWindow(QMainWindow):
    """
    Primary application window for TileVision AI.

    Provides the navigation sidebar and content stack.
    Manages the lifecycle of UI views and their associated ViewModels.
    """

    def __init__(
        self,
        indexing_viewmodel: IndexingViewModel,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Initialize the MainWindow.

        Args:
            indexing_viewmodel: Pre-configured IndexingViewModel instance.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._indexing_viewmodel = indexing_viewmodel

        self.setWindowTitle("TileVision AI — Visual Tile Search")
        self.setMinimumSize(1100, 760)
        self.resize(1280, 800)

        self._setup_ui()
        self._apply_styles()
        self._setup_status_bar()
        self._connect_signals()

        logger.info("MainWindow initialized and displayed.")

    # ── UI Construction ───────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        """Build the main window layout: sidebar + content stack."""
        # Central container
        central_widget = QWidget()
        central_widget.setObjectName("CentralWidget")
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Left Sidebar Navigation
        root_layout.addWidget(self._build_sidebar())

        # ── Vertical Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.VLine)
        divider.setObjectName("SidebarDivider")
        root_layout.addWidget(divider)

        # ── Content Stack
        self._content_stack = QStackedWidget()
        self._content_stack.setObjectName("ContentStack")
        root_layout.addWidget(self._content_stack, stretch=1)

        # ── Add Views to Content Stack
        self._indexing_view = IndexingView(self._indexing_viewmodel)
        self._content_stack.addWidget(self._indexing_view)  # index 0

        # Activate the first nav button by default
        self._nav_index_button.setChecked(True)
        self._content_stack.setCurrentIndex(0)

    def _build_sidebar(self) -> QFrame:
        """Build the left navigation sidebar."""
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(104)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(2, 16, 2, 16)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # ── App Logo / Brand
        brand_label = QLabel("TV\nAI")
        brand_label.setObjectName("BrandLabel")
        brand_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_font = QFont()
        brand_font.setPointSize(14)
        brand_font.setBold(True)
        brand_label.setFont(brand_font)
        layout.addWidget(brand_label)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setObjectName("SidebarSeparator")
        layout.addWidget(separator)
        layout.addSpacing(8)

        # ── Navigation Buttons
        self._nav_index_button = NavButton("📁", "Index")
        self._nav_index_button.clicked.connect(lambda: self._navigate(0))
        layout.addWidget(self._nav_index_button)

        # Placeholder nav buttons (for future features)
        self._nav_search_button = NavButton("🔍", "Search")
        self._nav_search_button.setEnabled(False)
        self._nav_search_button.setToolTip("Visual Search — Coming in Feature 2")
        layout.addWidget(self._nav_search_button)

        self._nav_catalog_button = NavButton("🗂", "Catalog")
        self._nav_catalog_button.setEnabled(False)
        self._nav_catalog_button.setToolTip("Tile Catalog — Coming in Feature 3")
        layout.addWidget(self._nav_catalog_button)

        self._nav_settings_button = NavButton("⚙️", "Settings")
        self._nav_settings_button.setEnabled(False)
        self._nav_settings_button.setToolTip("Settings — Coming soon")
        layout.addWidget(self._nav_settings_button)

        layout.addStretch()

        # ── Bottom: Version Label
        version_label = QLabel("v1.0.0")
        version_label.setObjectName("VersionLabel")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)

        return sidebar

    def _setup_status_bar(self) -> None:
        """Configure the bottom status bar."""
        status_bar = QStatusBar()
        status_bar.setObjectName("AppStatusBar")
        self.setStatusBar(status_bar)

        self._status_label = QLabel("Ready.")
        self._status_label.setObjectName("StatusBarLabel")
        status_bar.addWidget(self._status_label)

        # Right-side permanent label showing licensing info
        self._license_status_label = QLabel("🔐 Licensed")
        self._license_status_label.setObjectName("LicenseStatusLabel")
        status_bar.addPermanentWidget(self._license_status_label)

    # ── Signals ───────────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        """Connect ViewModel signals to the status bar and other global UI elements."""
        self._indexing_viewmodel.status_message.connect(self._update_status_bar)
        self._indexing_viewmodel.state_changed.connect(self._on_indexing_state_changed)

    # ── Navigation ────────────────────────────────────────────────────────────

    @Slot(int)
    def _navigate(self, index: int) -> None:
        """
        Switch the visible content to the requested stack index.

        Args:
            index: Stack widget page index.
        """
        # Uncheck all nav buttons
        for btn in [
            self._nav_index_button,
            self._nav_search_button,
            self._nav_catalog_button,
            self._nav_settings_button,
        ]:
            btn.setChecked(False)

        # Activate the appropriate button
        nav_map = {
            0: self._nav_index_button,
            1: self._nav_search_button,
            2: self._nav_catalog_button,
            3: self._nav_settings_button,
        }
        if index in nav_map:
            nav_map[index].setChecked(True)

        self._content_stack.setCurrentIndex(index)
        logger.debug(f"Navigated to content stack index: {index}")

    # ── Slots ─────────────────────────────────────────────────────────────────

    @Slot(str)
    def _update_status_bar(self, message: str) -> None:
        """
        Update the bottom status bar with an informational message.

        Args:
            message: Status text to display.
        """
        self._status_label.setText(message)

    @Slot(str)
    def _on_indexing_state_changed(self, state: str) -> None:
        """
        Update the window title to reflect the current indexing state.

        Args:
            state: Current IndexingState string.
        """
        state_display = {
            IndexingState.IDLE: "",
            IndexingState.RUNNING: " — Indexing...",
            IndexingState.PAUSED: " — Paused",
            IndexingState.FINISHED: " — Indexing Complete",
            IndexingState.CANCELLED: " — Cancelled",
            IndexingState.ERROR: " — Error",
        }
        suffix = state_display.get(state, "")
        self.setWindowTitle(f"TileVision AI — Visual Tile Search{suffix}")

    # ── Window Events ─────────────────────────────────────────────────────────

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Handle the window close event.

        Prompt the user if an indexing operation is currently running
        to prevent accidental data loss.

        Args:
            event: The Qt close event.
        """
        vm = self._indexing_viewmodel

        if vm.state in (IndexingState.RUNNING, IndexingState.PAUSED):
            reply = QMessageBox.question(
                self,
                "Indexing in Progress",
                "An indexing operation is currently active.\n\n"
                "Closing now will cancel it. Any tiles already processed will be saved.\n\n"
                "Are you sure you want to quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                logger.warning("User force-closed the application during active indexing. Cancelling worker.")
                vm.cancel_indexing()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    # ── Styling ───────────────────────────────────────────────────────────────

    def _apply_styles(self) -> None:
        """Apply global QSS styles to the main window."""
        self.setStyleSheet("""
            /* ── Base ──────────────────────────────────────────────────────── */
            QMainWindow {
                background-color: #1A1D26;
            }
            #CentralWidget {
                background-color: #1A1D26;
            }

            /* ── Sidebar ────────────────────────────────────────────────────  */
            #Sidebar {
                background-color: #13151F;
                border-right: none;
            }
            #SidebarDivider {
                color: #2D3250;
                border: none;
                border-left: 1px solid #2D3250;
                max-width: 1px;
            }
            #BrandLabel {
                color: #5C6BC0;
                font-size: 14px;
                font-weight: bold;
            }
            #SidebarSeparator {
                color: #2D3250;
                border: none;
                border-top: 1px solid #2D3250;
            }
            #VersionLabel {
                color: #37474F;
                font-size: 10px;
            }

            /* ── Nav Buttons ─────────────────────────────────────────────── */
            #NavButton {
                background-color: transparent;
                border: none;
                border-radius: 8px;
                color: #546E7A;
                text-align: center;
            }
            #NavButton:hover:!checked:enabled {
                background-color: #1E2130;
            }
            #NavButton:checked {
                background-color: #1E2847;
                border-left: 3px solid #5C6BC0;
            }
            #NavButton:disabled {
                opacity: 0.3;
            }
            #NavIcon {
                font-size: 18px;
                color: #546E7A;
            }
            #NavButton:checked #NavIcon {
                color: #7986CB;
            }
            #NavLabel {
                font-size: 10px;
                color: #546E7A;
            }
            #NavButton:checked #NavLabel {
                color: #7986CB;
                font-weight: bold;
            }

            /* ── Content Stack ───────────────────────────────────────────── */
            #ContentStack {
                background-color: #1A1D26;
            }

            /* ── Status Bar ──────────────────────────────────────────────── */
            QStatusBar {
                background-color: #13151F;
                border-top: 1px solid #2D3250;
            }
            #StatusBarLabel {
                color: #546E7A;
                font-size: 11px;
                padding-left: 8px;
            }
            #LicenseStatusLabel {
                color: #69F0AE;
                font-size: 11px;
                padding-right: 12px;
            }

            /* ── General QLabel ──────────────────────────────────────────── */
            QLabel {
                color: #E8EAF6;
                font-family: "Segoe UI", "Inter", sans-serif;
                font-size: 12px;
            }

            /* ── QToolTip ────────────────────────────────────────────────── */
            QToolTip {
                background-color: #252837;
                border: 1px solid #3D4166;
                color: #E8EAF6;
                font-size: 11px;
                padding: 4px 8px;
                border-radius: 4px;
            }
        """)
