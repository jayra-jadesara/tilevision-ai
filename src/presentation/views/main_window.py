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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from src.ai.feature_versions import FeatureVersionStatus

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QIcon, QAction, QCloseEvent, QPixmap
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
    QApplication,
)

from src.presentation.views.indexing_view import IndexingView
from src.presentation.viewmodels.indexing_viewmodel import IndexingViewModel, IndexingState
from src.presentation.views.search_view import SearchView
from src.presentation.viewmodels.search_viewmodel import SearchViewModel
from src.presentation.views.duplicates_view import DuplicatesView
from src.presentation.views.settings_view import SettingsView
from src.presentation.views.dashboard_view import DashboardView
from src.presentation.views.help_view import HelpView
from src.theme.theme_manager import adapt_legacy_stylesheet, get_app_stylesheet

logger = logging.getLogger("tilevision.presentation.views.main_window")

_RESOURCES_DIR = Path(__file__).resolve().parents[2] / "resources"
_LOGO_SMALL_PATH = _RESOURCES_DIR / "logo_small.png"
_APP_ICON_PATH = _RESOURCES_DIR / "app_icon.ico"


@dataclass
class DashboardDataProviders:
    """
    Bundles the extra data-source callables needed for the Professional
    Dashboard (Task A) into a single object, rather than growing
    MainWindow's constructor with 6+ individual provider parameters.
    All fields are optional; any left as None simply show "—"/empty on
    the Dashboard rather than erroring.
    """
    indexed_folder_count: Optional[Callable[[], int]] = None
    database_size: Optional[Callable[[], int]] = None
    faiss_size: Optional[Callable[[], int]] = None
    last_search: Optional[Callable[[], object]] = None
    recent_activity: Optional[Callable[[], List[object]]] = None
    recent_searches: Optional[Callable[[], List[object]]] = None


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
        search_viewmodel: Optional[SearchViewModel] = None,
        license_details: Optional[dict] = None,
        find_duplicates_use_case=None,
        settings=None,
        catalog_count_provider: Optional[Callable[[], int]] = None,
        dashboard_providers: Optional[DashboardDataProviders] = None,
        db_path_provider: Optional[Callable[[], object]] = None,
        indexing_use_case=None,
        indexed_folders_provider: Optional[Callable[[], List[str]]] = None,
        feature_version_provider: Optional[Callable[[], FeatureVersionStatus]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Initialize the MainWindow.

        Args:
            indexing_viewmodel: Pre-configured IndexingViewModel instance.
            search_viewmodel: Pre-configured SearchViewModel instance. If
                omitted, the Search nav item stays disabled (e.g. if the
                catalog has no images indexed yet is still a valid reason
                to show Search — only a missing viewmodel disables it).
            license_details: The dict returned by
                ValidateLicenseUseCase.verify_existing_license() at startup
                (customer_name/license_type/is_trial/days_remaining), used
                to populate the status bar's license indicator.
            find_duplicates_use_case: Pre-configured FindDuplicatesUseCase.
                If omitted, the Duplicates nav item stays disabled.
            settings: The shared AppSettings instance. If omitted, the
                Settings nav item stays disabled.
            catalog_count_provider: Callable returning the current number
                of indexed tiles, for the Settings/Dashboard Overview stats.
            dashboard_providers: Optional DashboardDataProviders bundle for
                the Professional Dashboard's extra stat cards / activity /
                search history panels (Task A). Any omitted field just
                shows "—"/empty rather than erroring.
            db_path_provider: Callable returning the SQLite database's
                absolute Path (Task D: Settings — Database Path display,
                Backup Database).
            indexing_use_case: The shared IndexImagesUseCase, needed for
                Settings' "Rebuild FAISS Index" action (Task D).
            indexed_folders_provider: Callable returning every folder
                that's been indexed, the set Rebuild FAISS Index operates
                over (Task D).
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._indexing_viewmodel = indexing_viewmodel
        self._search_viewmodel = search_viewmodel
        self._license_details = license_details or {}
        self._find_duplicates_use_case = find_duplicates_use_case
        self._settings = settings
        self._catalog_count_provider = catalog_count_provider
        self._dashboard_providers = dashboard_providers or DashboardDataProviders()
        self._db_path_provider = db_path_provider
        self._indexing_use_case = indexing_use_case
        self._indexed_folders_provider = indexed_folders_provider
        self._feature_version_provider = feature_version_provider
        self._current_theme = getattr(self._settings, "theme", "dark") if self._settings is not None else "dark"

        self.setWindowTitle("TileVision AI — Visual Tile Search")
        if _APP_ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(_APP_ICON_PATH)))
        self.setMinimumSize(1100, 760)

        # Open maximized by default
        self.showMaximized()

        self._setup_ui()
        self._apply_styles()
        self._setup_status_bar()
        self._connect_signals()
        self.refresh_stale_feature_banner()

        # Apply the app-level theme unconditionally
        self._on_theme_changed_request(self._current_theme)

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

        # ── Right content column (stale banner + stack)
        content_column = QWidget()
        content_column.setObjectName("ContentColumn")
        content_layout = QVBoxLayout(content_column)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self._stale_banner = QLabel()
        self._stale_banner.setObjectName("StaleFeatureBanner")
        self._stale_banner.setWordWrap(True)
        self._stale_banner.setVisible(False)
        self._stale_banner.setContentsMargins(12, 8, 12, 8)
        content_layout.addWidget(self._stale_banner)

        self._content_stack = QStackedWidget()
        self._content_stack.setObjectName("ContentStack")
        content_layout.addWidget(self._content_stack, stretch=1)

        root_layout.addWidget(content_column, stretch=1)

        # ── Add Views to Content Stack
        self._indexing_view = IndexingView(self._indexing_viewmodel, theme=self._current_theme)
        self._content_stack.addWidget(self._indexing_view)  # index 0

        if self._search_viewmodel is not None:
            self._search_view = SearchView(self._search_viewmodel, theme=self._current_theme)
            self._content_stack.addWidget(self._search_view)  # index 1
            self._nav_search_button.setEnabled(True)
            self._nav_search_button.setToolTip("Visual Search")
        else:
            # Placeholder page so the stack index still lines up with the
            # nav button map even when Search hasn't been wired up.
            self._content_stack.addWidget(QWidget())  # index 1

        if self._find_duplicates_use_case is not None:
            self._nav_duplicates_button.setEnabled(True)
            self._nav_duplicates_button.setToolTip("Duplicate Detection")

        # index 2: Dashboard (repurposes the "Catalog" nav slot)
        if self._catalog_count_provider is not None:
            self._dashboard_view = DashboardView(
                catalog_count_provider=self._catalog_count_provider,
                watched_folder_count_provider=(
                    lambda: len(self._settings.watch_folders) if self._settings else 0
                ),
                indexed_folder_count_provider=self._dashboard_providers.indexed_folder_count,
                database_size_provider=self._dashboard_providers.database_size,
                faiss_size_provider=self._dashboard_providers.faiss_size,
                last_search_provider=self._dashboard_providers.last_search,
                recent_activity_provider=self._dashboard_providers.recent_activity,
                recent_searches_provider=self._dashboard_providers.recent_searches,
                license_details=self._license_details,
                on_go_to_index=lambda: self._navigate(0),
                on_go_to_search=lambda: self._navigate(1),
                on_go_to_duplicates=self._on_duplicates_clicked if self._find_duplicates_use_case else None,
                on_go_to_settings=(lambda: self._navigate(3)) if self._settings is not None else None,
                on_repeat_search=self._on_dashboard_repeat_search,
                theme=self._current_theme,
            )
            self._content_stack.addWidget(self._dashboard_view)  # index 2
            self._nav_catalog_button.setEnabled(True)
            self._nav_catalog_button.setToolTip("Dashboard")
            self._nav_catalog_button.clicked.connect(lambda: self._navigate(2))
        else:
            self._content_stack.addWidget(QWidget())  # index 2

        if self._settings is not None:
            self._settings_view = SettingsView(
                settings=self._settings,
                license_details=self._license_details,
                catalog_count_provider=self._catalog_count_provider,
                on_theme_changed=self._on_theme_changed_request,
                db_path_provider=self._db_path_provider,
                indexing_use_case=self._indexing_use_case,
                indexed_folders_provider=self._indexed_folders_provider,
                theme=self._current_theme,
            )
            self._content_stack.addWidget(self._settings_view)  # index 3
            self._nav_settings_button.setEnabled(True)
            self._nav_settings_button.setToolTip("Settings")
            self._nav_settings_button.clicked.connect(lambda: self._navigate(3))
        else:
            self._content_stack.addWidget(QWidget())  # index 3

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
        logo_label = QLabel()
        logo_pixmap = QPixmap(str(_LOGO_SMALL_PATH))
        if not logo_pixmap.isNull():
            logo_label.setPixmap(
                logo_pixmap.scaled(
                    56, 56, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
            )
        else:
            # Fallback if the logo asset is ever missing — never crash on a
            # missing image, just fall back to the original text mark.
            logo_label.setText("TV\nAI")
            logo_font = QFont()
            logo_font.setPointSize(14)
            logo_font.setBold(True)
            logo_label.setFont(logo_font)
        logo_label.setObjectName("BrandLabel")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo_label)

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
        self._nav_search_button.clicked.connect(lambda: self._navigate(1))
        layout.addWidget(self._nav_search_button)

        self._nav_duplicates_button = NavButton("🧬", "Duplicates")
        self._nav_duplicates_button.setEnabled(False)
        self._nav_duplicates_button.setToolTip("Duplicate Detection")
        self._nav_duplicates_button.clicked.connect(self._on_duplicates_clicked)
        layout.addWidget(self._nav_duplicates_button)

        self._nav_catalog_button = NavButton("🏠", "Dashboard")
        self._nav_catalog_button.setEnabled(False)
        self._nav_catalog_button.setToolTip("Dashboard")
        layout.addWidget(self._nav_catalog_button)

        self._nav_settings_button = NavButton("⚙️", "Settings")
        self._nav_settings_button.setEnabled(False)
        self._nav_settings_button.setToolTip("Settings — Coming soon")
        layout.addWidget(self._nav_settings_button)

        self._nav_help_button = NavButton("❓", "Help")
        self._nav_help_button.setToolTip("Help & User Guide")
        self._nav_help_button.setCheckable(False)  # opens a modal, not a nav page
        self._nav_help_button.clicked.connect(self._on_help_clicked)
        layout.addWidget(self._nav_help_button)

        layout.addStretch()

        # ── Bottom: Version Label (credits JD Software via tooltip — the
        # sidebar is too narrow to show the full credit line as text; the
        # complete "Made by JD Software" + contact number also appears in
        # the Help dialog's footer for anyone who wants it front-and-center)
        version_label = QLabel("v1.0.0")
        version_label.setObjectName("VersionLabel")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setToolTip("TileVision AI v1.0.0\nMade by JD Software\nContact: 88662 77767")
        version_label.setCursor(Qt.CursorShape.PointingHandCursor)
        version_label.mousePressEvent = lambda event: self._on_help_clicked()
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
        self._license_status_label = QLabel(self._format_license_status())
        self._license_status_label.setObjectName("LicenseStatusLabel")
        status_bar.addPermanentWidget(self._license_status_label)

    def _format_license_status(self) -> str:
        """Build the status bar license indicator text from license_details."""
        if not self._license_details:
            return "🔓 Unlicensed"

        if self._license_details.get("is_trial"):
            days = self._license_details.get("days_remaining", 0)
            if days <= 3:
                return f"⏳ Trial: {days} day(s) left"
            return f"🕐 Trial: {days} days left"

        license_type = self._license_details.get("license_type", "Licensed")
        return f"🔐 {license_type}"

    # ── Signals ───────────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        """Connect ViewModel signals to the status bar and other global UI elements."""
        self._indexing_viewmodel.status_message.connect(self._update_status_bar)
        self._indexing_viewmodel.state_changed.connect(self._on_indexing_state_changed)
        self._indexing_viewmodel.indexing_completed.connect(self._on_indexing_completed)

        if self._search_viewmodel is not None:
            self._search_viewmodel.status_message.connect(self._update_status_bar)

    @Slot(object)
    def _on_indexing_completed(self, result) -> None:
        """Refresh search filter dropdowns after catalog changes."""
        if self._search_viewmodel is not None and result.is_completed:
            self._search_viewmodel.load_filter_options()
        self.refresh_stale_feature_banner()

    def refresh_stale_feature_banner(self) -> None:
        """Show or hide the stale-feature warning banner."""
        if self._feature_version_provider is None:
            self._stale_banner.setVisible(False)
            return

        try:
            status = self._feature_version_provider()
        except Exception as exc:
            logger.warning("Failed to read feature version status: %s", exc)
            self._stale_banner.setVisible(False)
            return

        if status.is_compatible or status.stale_count <= 0:
            self._stale_banner.setVisible(False)
            return

        self._stale_banner.setText(
            "⚠️ Indexed features are outdated "
            f"({status.stale_count} of {status.indexed_count} tiles). "
            "Re-scan your folders or use Settings → Rebuild FAISS Index "
            "for accurate search results."
        )
        self._stale_banner.setVisible(True)

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

        if index == 2 and hasattr(self, "_dashboard_view"):
            self._dashboard_view.refresh()

    def _on_duplicates_clicked(self) -> None:
        """Open the Duplicate Detection dialog (modal, like License activation)."""
        if self._find_duplicates_use_case is None:
            return
        dialog = DuplicatesView(self._find_duplicates_use_case, parent=self, theme=self._current_theme)
        dialog.exec()

    def _on_dashboard_repeat_search(self, query_image_path: str) -> None:
        """Navigate to Search and re-run a query from the Dashboard's Recent Searches panel."""
        if self._search_viewmodel is None:
            return
        self._navigate(1)
        self._search_viewmodel.repeat_search(query_image_path)

    def _on_help_clicked(self) -> None:
        """Open the Help & User Guide dialog (Task E)."""
        dialog = HelpView(parent=self, theme=self._current_theme)
        dialog.exec()

    def _on_theme_changed_request(self, theme: str) -> None:
        """
        Apply the new theme app-wide: the MainWindow chrome (via the
        QApplication-level stylesheet) plus every currently-open view,
        each of which now builds its own stylesheet from the shared
        palette (see theme_manager.get_palette) and exposes set_theme()
        for exactly this purpose.
        """
        self._current_theme = theme
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(get_app_stylesheet(theme))
        self._apply_styles()

        for view_attr in ("_indexing_view", "_search_view", "_dashboard_view", "_settings_view"):
            view = getattr(self, view_attr, None)
            if view is not None and hasattr(view, "set_theme"):
                view.set_theme(theme)

        logger.info(f"Applied '{theme}' theme.")

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
            IndexingState.CANCELLING: " — Cancelling...",
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

        if vm.state in (IndexingState.RUNNING, IndexingState.PAUSED, IndexingState.CANCELLING):
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
        self.setStyleSheet(adapt_legacy_stylesheet("""
            /* ── Base ──────────────────────────────────────────────────────── */
            QMainWindow {
                background-color: #1A1D26;
            }
            #CentralWidget {
                background-color: #1A1D26;
            }
            #StaleFeatureBanner {
                background-color: #453410;
                color: #FBBF24;
                padding: 8px 12px;
                border-bottom: 1px solid #5C4A1A;
                font-size: 12px;
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
        """, self._current_theme))
