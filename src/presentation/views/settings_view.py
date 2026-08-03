"""
Settings View for TileVision AI (Task D: Settings).

Lets the user configure:
  - Theme (dark/light), thumbnail size, number of search results.
  - Watched folders for auto-indexing (Feature 7) — add/remove.
  - Language (placeholder — English only for now).
  - Backup Database (uses the database path internally; not shown to the
    user — an absolute file path invites accidental navigation/deletion).
  - Rebuild FAISS Index (force re-embed everything).
  - Clear thumbnail Cache.
  - Export Logs.
"""

import logging
import shutil
from pathlib import Path
from typing import Callable, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QComboBox,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QProgressDialog,
    QTabWidget,
    QScrollArea,
    QFrame,
    QGridLayout,
    QLineEdit,
    QCheckBox,
)

from src.ai.feature_versions import CURRENT_FEATURE_VERSION, FeatureVersionStatus
from src.ai.gpu_info import GpuRuntimeInfo
from src.config.settings import AppSettings
from src.core.use_cases.index_images import IndexImagesUseCase
from src.core.use_cases.monitor_folder import is_watchdog_available
from src.licensing.hardware import get_machine_fingerprint
from src.presentation.workers.rebuild_index_worker import RebuildIndexWorker
from src.utils.logger import get_log_file_path
from src.presentation.views.catalogue_profiles_panel import CatalogueProfilesPanel
from src.theme.theme_manager import get_shared_view_qss, get_settings_view_qss

logger = logging.getLogger("tilevision.presentation.views.settings_view")


class SettingsView(QWidget):
    """Settings page widget."""

    def __init__(
        self,
        settings: AppSettings,
        license_details: Optional[dict] = None,
        catalogue_master_service=None,
        catalog_count_provider: Optional[Callable[[], int]] = None,
        on_theme_changed: Optional[Callable[[str], None]] = None,
        db_path_provider: Optional[Callable[[], Path]] = None,
        indexing_use_case: Optional[IndexImagesUseCase] = None,
        indexed_folders_provider: Optional[Callable[[], List[str]]] = None,
        on_catalog_changed: Optional[Callable[[], None]] = None,
        on_watch_folders_changed: Optional[Callable[[], None]] = None,
        on_check_updates: Optional[Callable[[], None]] = None,
        feature_version_provider: Optional[Callable[[], FeatureVersionStatus]] = None,
        gpu_info_provider: Optional[Callable[[], GpuRuntimeInfo]] = None,
        diagnostics_info_provider: Optional[Callable[[], dict]] = None,
        search_optimization_engine=None,
        vector_index=None,
        theme: str = "dark",
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Args:
            settings: The shared AppSettings instance (read/write).
            license_details: Current license/trial info for display.
            catalog_count_provider: Callable returning the current number
                of indexed tiles, for the "Catalog" stat.
            on_theme_changed: Callback invoked with the new theme name
                ("dark"/"light") when the user changes the theme dropdown.
            db_path_provider: Callable returning the SQLite database's
                absolute Path, used internally for the Backup Database
                action. If omitted, that control is disabled.
            indexing_use_case: The shared IndexImagesUseCase, needed for
                the "Rebuild FAISS Index" action (Task D). If omitted,
                that button is disabled.
            indexed_folders_provider: Callable returning every folder
                that's been indexed at least once — the set Rebuild FAISS
                Index operates over. If omitted, Rebuild is disabled.
            theme: Initial theme ("dark"/"light") to render with.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._theme = theme
        self._settings = settings
        self._license_details = license_details or {}
        self._catalogue_master_service = catalogue_master_service
        self._catalog_count_provider = catalog_count_provider
        self._on_theme_changed = on_theme_changed
        self._db_path_provider = db_path_provider
        self._indexing_use_case = indexing_use_case
        self._indexed_folders_provider = indexed_folders_provider
        self._on_catalog_changed = on_catalog_changed
        self._on_watch_folders_changed = on_watch_folders_changed
        self._on_check_updates = on_check_updates
        self._feature_version_provider = feature_version_provider
        self._gpu_info_provider = gpu_info_provider
        self._diagnostics_info_provider = diagnostics_info_provider
        self._search_optimization_engine = search_optimization_engine
        self._vector_index = vector_index
        self._rebuild_worker: Optional[RebuildIndexWorker] = None
        self._rebuild_progress_dialog: Optional[QProgressDialog] = None
        self._setup_ui()
        self._apply_styles()
        self.refresh_feature_status()

    def refresh_feature_status(self) -> None:
        """Update overview stat cards that depend on live catalog data."""
        if hasattr(self, "_tiles_count_label") and self._catalog_count_provider is not None:
            try:
                self._tiles_count_label.setText(str(self._catalog_count_provider()))
            except Exception as exc:
                logger.warning("Failed to read catalog count: %s", exc)

        if self._feature_version_provider is None:
            self._feature_status_label.setText("—")
        else:
            try:
                status = self._feature_version_provider()
            except Exception as exc:
                logger.warning("Failed to read feature version status: %s", exc)
                self._feature_status_label.setText("Unknown")
            else:
                if status.indexed_count == 0:
                    self._feature_status_label.setText("No tiles indexed yet")
                elif status.is_compatible:
                    self._feature_status_label.setText(
                        f"Up to date (v{CURRENT_FEATURE_VERSION}, {status.indexed_count} tiles)"
                    )
                else:
                    self._feature_status_label.setText(
                        f"Outdated — {status.stale_count} of {status.indexed_count} tiles "
                        f"need re-index (v{CURRENT_FEATURE_VERSION})"
                    )

        if hasattr(self, "_search_engine_status_label"):
            self.refresh_search_engine_status()

    def _gpu_summary_text(self) -> str:
        if self._gpu_info_provider is None:
            return "—"
        try:
            return self._gpu_info_provider().summary_for_ui()
        except Exception as exc:
            logger.warning("Failed to read GPU status: %s", exc)
            return "Unknown"

    def _setup_ui(self) -> None:
        self.setObjectName("SettingsView")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title = QLabel("Settings")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "Configure search preferences, auto folder monitoring, maintenance tools, "
            "and export catalogue profiles."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        self._tabs = QTabWidget()
        self._tabs.setObjectName("SettingsTabs")

        general_page = QWidget()
        general_page.setObjectName("SettingsGeneralPage")
        general_layout = QVBoxLayout(general_page)
        general_layout.setContentsMargins(4, 8, 4, 8)
        general_layout.setSpacing(20)
        general_layout.addWidget(self._build_overview_row())
        general_layout.addWidget(self._build_machine_id_section())
        general_layout.addWidget(self._build_watched_folders_section())

        columns = QHBoxLayout()
        columns.setSpacing(16)
        columns.addWidget(self._build_preferences_section(), stretch=1)
        columns.addWidget(self._build_maintenance_section(), stretch=1)
        general_layout.addLayout(columns)
        general_layout.addWidget(self._build_search_engine_section())
        general_layout.addStretch()

        general_scroll = QScrollArea()
        general_scroll.setObjectName("SettingsGeneralScroll")
        general_scroll.setWidgetResizable(True)
        general_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        general_scroll.setWidget(general_page)

        self._export_profiles_panel = CatalogueProfilesPanel(
            theme=self._theme,
            catalogue_master_service=self._catalogue_master_service,
            license_customer_name=str(self._license_details.get("customer_name") or ""),
        )
        self._tabs.addTab(general_scroll, "General")
        self._tabs.addTab(self._export_profiles_panel, "Export Profiles")
        layout.addWidget(self._tabs, stretch=1)

    def show_export_profiles_tab(self) -> None:
        """Switch to the Export Profiles tab (called from Export Catalogue)."""
        index = self._tabs.indexOf(self._export_profiles_panel)
        if index >= 0:
            self._tabs.setCurrentIndex(index)

    def _make_stat_card(self, title: str, value: str) -> tuple[QFrame, QLabel]:
        card = QFrame()
        card.setObjectName("StatCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 12, 14, 12)
        card_layout.setSpacing(4)

        title_label = QLabel(title.upper())
        title_label.setObjectName("StatCardTitle")
        value_label = QLabel(value)
        value_label.setObjectName("StatCardValue")
        value_label.setWordWrap(True)

        card_layout.addWidget(title_label)
        card_layout.addWidget(value_label)
        return card, value_label

    def _build_overview_row(self) -> QWidget:
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)

        catalog_count = self._catalog_count_provider() if self._catalog_count_provider else "—"
        tiles_card, self._tiles_count_label = self._make_stat_card("Indexed Tiles", str(catalog_count))
        row.addWidget(tiles_card, stretch=1)

        feature_card, self._feature_status_label = self._make_stat_card("Feature Index", "—")
        row.addWidget(feature_card, stretch=1)

        gpu_card, self._gpu_status_label = self._make_stat_card(
            "AI Device", self._gpu_summary_text()
        )
        row.addWidget(gpu_card, stretch=1)

        if self._license_details.get("is_trial"):
            days = self._license_details.get("days_remaining", 0)
            license_text = f"Trial · {days} day(s) left"
        elif self._license_details:
            license_text = (
                f"{self._license_details.get('license_type', 'Licensed')} · "
                f"{self._license_details.get('customer_name', '')}"
            )
        else:
            license_text = "Unlicensed"
        license_card, _ = self._make_stat_card("License", license_text)
        row.addWidget(license_card, stretch=1)

        return row_widget

    def _section_box(self, title: str) -> QGroupBox:
        box = QGroupBox(title)
        box.setObjectName("SettingsSection")
        return box

    def _form_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("SettingsFormLabel")
        return label

    def _build_machine_id_section(self) -> QGroupBox:
        box = self._section_box("Machine ID")
        layout = QVBoxLayout(box)

        note = QLabel(
            "Send this Machine ID to your TileVision vendor when requesting a license key. "
            "It uniquely identifies this computer."
        )
        note.setObjectName("SectionNote")
        note.setWordWrap(True)
        layout.addWidget(note)

        row = QHBoxLayout()
        row.setSpacing(10)
        self._machine_id_edit = QLineEdit()
        self._machine_id_edit.setObjectName("MachineIdEdit")
        self._machine_id_edit.setReadOnly(True)
        self._machine_id_edit.setPlaceholderText("Loading Machine ID...")
        try:
            self._machine_id_edit.setText(get_machine_fingerprint())
        except Exception as exc:
            logger.warning("Failed to read machine fingerprint: %s", exc)
            self._machine_id_edit.setText("")

        copy_button = QPushButton("Copy Machine ID")
        copy_button.setObjectName("SecondaryButton")
        copy_button.clicked.connect(self._on_copy_machine_id)
        row.addWidget(self._machine_id_edit, stretch=1)
        row.addWidget(copy_button)
        layout.addLayout(row)
        return box

    def _on_copy_machine_id(self) -> None:
        machine_id = self._machine_id_edit.text().strip()
        if not machine_id:
            QMessageBox.warning(self, "Unavailable", "Machine ID could not be read on this PC.")
            return
        QGuiApplication.clipboard().setText(machine_id)
        QMessageBox.information(
            self,
            "Copied",
            "Machine ID copied to clipboard. Send it to your vendor to receive a license key.",
        )

    def _build_watched_folders_section(self) -> QGroupBox:
        box = self._section_box("Auto Folder Monitoring")
        layout = QVBoxLayout(box)

        note = QLabel(
            "Folders listed here are watched automatically — new, changed, or deleted "
            "images are indexed in the background without a manual scan. Changes apply "
            "immediately while the app is running."
        )
        note.setObjectName("SectionNote")
        note.setWordWrap(True)
        layout.addWidget(note)

        self._watchdog_warning = QLabel(
            "Folder monitoring requires the watchdog package, which is not installed. "
            "Run pip install watchdog in your environment, then restart TileVision AI."
        )
        self._watchdog_warning.setObjectName("WatchdogWarning")
        self._watchdog_warning.setWordWrap(True)
        self._watchdog_warning.setVisible(not is_watchdog_available())
        layout.addWidget(self._watchdog_warning)

        self._folders_list = QListWidget()
        self._folders_list.setObjectName("FoldersList")
        for folder in self._settings.watch_folders:
            self._folders_list.addItem(QListWidgetItem(folder))
        layout.addWidget(self._folders_list)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        add_button = QPushButton("Add Folder")
        add_button.setObjectName("SecondaryButton")
        add_button.clicked.connect(self._on_add_folder)
        button_row.addWidget(add_button)

        remove_button = QPushButton("Remove Selected")
        remove_button.setObjectName("SecondaryButton")
        remove_button.clicked.connect(self._on_remove_folder)
        button_row.addWidget(remove_button)
        button_row.addStretch()
        layout.addLayout(button_row)

        return box

    def _build_preferences_section(self) -> QGroupBox:
        box = self._section_box("Preferences")
        form = QFormLayout(box)
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self._top_k_combo = QComboBox()
        _RESULT_COUNT_OPTIONS = ["5", "10", "15", "20", "25"]
        self._top_k_combo.addItems(_RESULT_COUNT_OPTIONS)
        current_top_k = str(self._settings.top_k)
        idx = self._top_k_combo.findText(current_top_k)
        if idx < 0:
            # Current value isn't one of the presets (e.g. from an older
            # free-form setting) — add it so the dropdown still reflects
            # the real, active value rather than silently changing it.
            self._top_k_combo.insertItem(0, current_top_k)
            idx = 0
        self._top_k_combo.setCurrentIndex(idx)
        self._top_k_combo.currentTextChanged.connect(self._on_top_k_changed)
        form.addRow(self._form_label("Search Results"), self._top_k_combo)

        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["dark", "light"])
        current_theme = getattr(self._settings, "theme", "dark")
        idx = self._theme_combo.findText(current_theme)
        self._theme_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._theme_combo.currentTextChanged.connect(self._on_theme_selected)
        form.addRow(self._form_label("Theme"), self._theme_combo)

        self._language_combo = QComboBox()
        self._language_combo.addItem("English")
        self._language_combo.setEnabled(False)
        self._language_combo.setToolTip("Additional languages coming in a future release.")
        form.addRow(self._form_label("Language"), self._language_combo)

        self._sam2_checkbox = QCheckBox(
            "Use SAM 2 for Precise Crop (same on Windows / Mac Intel / Silicon) — default ON"
        )
        self._sam2_checkbox.setChecked(bool(self._settings.enable_sam2_precise_crop))
        self._sam2_checkbox.setToolTip(
            "Default ON. Precise Crop uses the same ONNX SAM2 path on Windows, "
            "Mac Intel, and Mac Apple Silicon, then GrabCut if needed. "
            "Default drop-to-search stays on fast OpenCV."
        )
        self._sam2_checkbox.toggled.connect(self._on_sam2_toggled)
        form.addRow(self._form_label("Precise Crop"), self._sam2_checkbox)

        self._sam2_status_label = QLabel()
        self._sam2_status_label.setObjectName("SectionNote")
        self._sam2_status_label.setWordWrap(True)
        self._refresh_sam2_status()
        form.addRow("", self._sam2_status_label)

        return box

    def _build_search_engine_section(self) -> QGroupBox:
        """Customer-facing Search Engine status (no FAISS backend jargon)."""
        from src.ai.search_optimization_engine import format_tile_count, is_developer_mode

        box = self._section_box("Search Engine")
        layout = QVBoxLayout(box)
        layout.setSpacing(10)

        self._search_engine_status_label = QLabel()
        self._search_engine_status_label.setObjectName("SectionNote")
        self._search_engine_status_label.setWordWrap(True)
        self._search_engine_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._search_engine_status_label)

        self._search_engine_summary_label = QLabel()
        self._search_engine_summary_label.setObjectName("SectionNote")
        self._search_engine_summary_label.setWordWrap(True)
        layout.addWidget(self._search_engine_summary_label)

        # Developer Mode only — hidden for production customers.
        self._dev_search_box = QGroupBox("Developer Mode — Search Diagnostics")
        self._dev_search_box.setObjectName("SettingsSection")
        dev_layout = QVBoxLayout(self._dev_search_box)
        self._dev_search_diag_label = QLabel()
        self._dev_search_diag_label.setObjectName("SectionNote")
        self._dev_search_diag_label.setWordWrap(True)
        self._dev_search_diag_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        dev_layout.addWidget(self._dev_search_diag_label)

        row = QHBoxLayout()
        self._dev_backend_combo = QComboBox()
        from src.ai.index_backends import IndexBackend, backend_display_name

        for value in (
            IndexBackend.FLAT_IP.value,
            IndexBackend.HNSW.value,
            IndexBackend.IVF.value,
            IndexBackend.IVF_PQ.value,
        ):
            self._dev_backend_combo.addItem(
                backend_display_name(IndexBackend.parse(value)), value
            )
        row.addWidget(QLabel("Backend override:"))
        row.addWidget(self._dev_backend_combo, stretch=1)
        self._dev_apply_backend_button = QPushButton("Apply Backend (Rebuild Required)")
        self._dev_apply_backend_button.setObjectName("SecondaryButton")
        self._dev_apply_backend_button.clicked.connect(self._on_dev_apply_backend)
        row.addWidget(self._dev_apply_backend_button)
        self._dev_reanalyze_button = QPushButton("Re-run Optimization Analysis")
        self._dev_reanalyze_button.setObjectName("SecondaryButton")
        self._dev_reanalyze_button.clicked.connect(
            lambda: self.refresh_search_engine_status(run_analysis=True)
        )
        row.addWidget(self._dev_reanalyze_button)
        dev_layout.addLayout(row)
        layout.addWidget(self._dev_search_box)
        self._dev_search_box.setVisible(is_developer_mode())

        self.refresh_search_engine_status(run_analysis=False)
        return box

    def refresh_search_engine_status(self, *, run_analysis: bool = False) -> None:
        """Refresh customer Search Engine panel from live catalog + SOE."""
        from src.ai.search_optimization_engine import format_tile_count

        catalog_size = 0
        try:
            if self._catalog_count_provider is not None:
                catalog_size = int(self._catalog_count_provider() or 0)
        except Exception as exc:
            logger.warning("Could not read catalog size for Search Engine: %s", exc)

        decision = None
        engine = getattr(self, "_search_optimization_engine", None)
        if run_analysis and engine is not None:
            try:
                decision = engine.analyze_and_decide(
                    catalog_size=catalog_size,
                    current_backend=self._settings.index_backend,
                    run_benchmark=True,
                )
                engine.apply_to_settings(self._settings, decision)
                vector_index = getattr(self, "_vector_index", None)
                if vector_index is not None and hasattr(vector_index, "configure_backend"):
                    vector_index.configure_backend(decision.selected_backend)
            except Exception as exc:
                logger.warning("Search optimization analysis failed: %s", exc)

        status = self._settings.search_optimization_status
        if status.lower().startswith("optimized"):
            status_line = "✓ Optimized"
        elif "need" in status.lower():
            status_line = "Needs Optimization"
        else:
            status_line = status

        last_opt = self._settings.last_search_optimization_at or "Not run yet"
        mode = (self._settings.search_engine_mode or "automatic").title()
        health = self._settings.search_health or "Excellent"
        index_status = self._settings.index_health_status or "Healthy"

        self._search_engine_status_label.setText(
            f"Status\n{status_line}\n\n"
            f"Mode\n{mode}\n\n"
            f"Catalog\n{format_tile_count(catalog_size)}\n\n"
            f"Search Health\n{health}\n\n"
            f"Index Status\n{index_status}\n\n"
            f"Last Optimization\n{last_opt}"
        )
        summary = self._settings.last_optimization_summary or (
            "TileVision automatically keeps search fast and accurate for this computer."
        )
        self._search_engine_summary_label.setText(summary)

        if getattr(self, "_dev_search_box", None) is not None and self._dev_search_box.isVisible():
            diag = ""
            if decision is not None:
                diag = decision.technical_reason + "\n\n" + str(decision.diagnostics)
            elif engine is not None and engine.last_decision is not None:
                last = engine.last_decision
                diag = last.technical_reason + "\n\n" + str(last.diagnostics)
            else:
                diag = (
                    f"configured_backend={self._settings.index_backend}\n"
                    f"catalog={catalog_size}"
                )
            self._dev_search_diag_label.setText(diag)
            idx = self._dev_backend_combo.findData(self._settings.index_backend)
            if idx >= 0:
                self._dev_backend_combo.setCurrentIndex(idx)

    def _on_dev_apply_backend(self) -> None:
        from src.ai.search_optimization_engine import is_developer_mode

        if not is_developer_mode():
            return
        value = self._dev_backend_combo.currentData()
        if not value:
            return
        self._settings.index_backend = str(value)
        vector_index = getattr(self, "_vector_index", None)
        if vector_index is not None and hasattr(vector_index, "configure_backend"):
            vector_index.configure_backend(str(value))
        self._settings.search_optimization_status = "Needs Optimization"
        self._settings.index_health_status = "Needs rebuild"
        self.refresh_search_engine_status(run_analysis=False)
        QMessageBox.information(
            self,
            "Developer Mode",
            "Backend override saved. Use Rebuild FAISS Index to apply on disk.",
        )

    def refresh_index_advisor(self) -> None:
        """Compatibility alias — refreshes Search Engine status."""
        self.refresh_search_engine_status(run_analysis=False)

    def start_automatic_search_optimization(self, *, reason: str = "") -> None:
        """
        Customer-safe automatic rebuild when SOE requires optimization.

        Progress copy never names FAISS backends.
        """
        if getattr(self, "_auto_optimize_running", False):
            return
        if self._indexing_use_case is None or self._indexed_folders_provider is None:
            return
        folders = self._indexed_folders_provider()
        if not folders:
            return
        self._auto_optimize_running = True
        self._settings.search_optimization_status = "Optimizing"
        self.refresh_search_engine_status(run_analysis=False)
        self._start_rebuild(
            skip_confirm=True,
            window_title="Optimizing Search",
            preparing_text="Optimizing Search...",
            progress_prefix="Rebuilding Search Index",
        )

    def _refresh_sam2_status(self) -> None:
        """
        Show Precise Crop readiness without importing onnxruntime.

        Importing ORT during MainWindow construction has contended with torch
        on some Mac installs and left Search stuck on "Searching...".
        Weights presence is enough for status; ORT loads on first Precise Crop.
        """
        try:
            from src.ai.preprocess.sam2_onnx_backend import (
                resolve_sam2_onnx_dir,
                sam2_onnx_enabled,
            )

            if not sam2_onnx_enabled():
                status = "Disabled in Settings"
            elif resolve_sam2_onnx_dir() is not None:
                status = "ONNX weights ready (loads on first Precise Crop)"
            else:
                status = (
                    "ONNX weights missing — reinstall v1.0.13+ or run "
                    "scripts/download_sam2_onnx_model.py"
                )
            self._sam2_status_label.setText(
                f"ONNX SAM2: {status}\n"
                "Default Search does not use SAM2 (fast OpenCV only)."
            )
        except Exception as exc:
            self._sam2_status_label.setText(f"Status unavailable: {exc}")

    def _on_sam2_toggled(self, enabled: bool) -> None:
        self._settings.enable_sam2_precise_crop = enabled
        try:
            from src.ai.preprocess.sam2_backend import configure_sam2_from_settings

            configure_sam2_from_settings(enabled)
        except Exception as exc:
            logger.warning("Could not apply SAM2 setting: %s", exc)
        self._refresh_sam2_status()

    def _build_maintenance_section(self) -> QGroupBox:
        box = self._section_box("Maintenance")
        layout = QVBoxLayout(box)

        note = QLabel(
            "Rebuild FAISS Index re-analyzes every tile after a software update. "
            "Clear Cache removes thumbnails only (they regenerate automatically). "
            "Backup Database, Export Logs, and Export Diagnostics are optional "
            "support tools — use them before major changes or when contacting support."
        )
        note.setObjectName("SectionNote")
        note.setWordWrap(True)
        layout.addWidget(note)

        grid = QGridLayout()
        grid.setSpacing(10)

        self._backup_button = QPushButton("Backup Database")
        self._backup_button.setObjectName("SecondaryButton")
        self._backup_button.clicked.connect(self._on_backup_database)
        self._backup_button.setEnabled(self._db_path_provider is not None)

        self._export_logs_button = QPushButton("Export Logs")
        self._export_logs_button.setObjectName("SecondaryButton")
        self._export_logs_button.clicked.connect(self._on_export_logs)

        self._rebuild_button = QPushButton("Rebuild FAISS Index")
        self._rebuild_button.setObjectName("SecondaryButton")
        self._rebuild_button.clicked.connect(self._on_rebuild_faiss)
        self._rebuild_button.setEnabled(
            self._indexing_use_case is not None and self._indexed_folders_provider is not None
        )

        self._clear_cache_button = QPushButton("Clear Cache")
        self._clear_cache_button.setObjectName("SecondaryButton")
        self._clear_cache_button.clicked.connect(self._on_clear_cache)

        self._export_diagnostics_button = QPushButton("Export Diagnostics")
        self._export_diagnostics_button.setObjectName("SecondaryButton")
        self._export_diagnostics_button.clicked.connect(self._on_export_diagnostics)

        for button in (
            self._backup_button,
            self._export_logs_button,
            self._rebuild_button,
            self._clear_cache_button,
            self._export_diagnostics_button,
        ):
            button.setMinimumHeight(36)

        grid.addWidget(self._backup_button, 0, 0)
        grid.addWidget(self._export_logs_button, 0, 1)
        grid.addWidget(self._rebuild_button, 1, 0)
        grid.addWidget(self._clear_cache_button, 1, 1)
        grid.addWidget(self._export_diagnostics_button, 2, 0, 1, 2)

        self._check_updates_button = QPushButton("Check for Updates")
        self._check_updates_button.setObjectName("SecondaryButton")
        self._check_updates_button.setMinimumHeight(36)
        self._check_updates_button.clicked.connect(self._on_check_updates_clicked)
        self._check_updates_button.setEnabled(self._on_check_updates is not None)
        grid.addWidget(self._check_updates_button, 3, 0, 1, 2)

        self._auto_update_checkbox = QCheckBox("Notify me when a new version is available")
        self._auto_update_checkbox.setChecked(self._settings.check_for_updates)
        self._auto_update_checkbox.toggled.connect(self._on_auto_update_toggled)
        layout.addWidget(self._auto_update_checkbox)

        layout.addLayout(grid)

        return box

    def _on_check_updates_clicked(self) -> None:
        if self._on_check_updates is not None:
            self._on_check_updates()

    def _on_auto_update_toggled(self, enabled: bool) -> None:
        self._settings.check_for_updates = enabled

    # ── Handlers: Watched Folders ────────────────────────────────────────

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Watch")
        if not folder:
            return

        current = self._settings.watch_folders
        resolved = str(Path(folder).resolve())
        if resolved in current:
            QMessageBox.information(self, "Already Added", "This folder is already being watched.")
            return

        current.append(resolved)
        self._settings.watch_folders = current
        self._folders_list.addItem(QListWidgetItem(resolved))
        logger.info(f"Added watched folder: {resolved}")
        self._apply_watch_folder_changes()

    def _apply_watch_folder_changes(self) -> None:
        self._watchdog_warning.setVisible(not is_watchdog_available())
        if self._on_watch_folders_changed is not None:
            self._on_watch_folders_changed()
        if self._settings.watch_folders and not is_watchdog_available():
            QMessageBox.warning(
                self,
                "Folder Monitoring Unavailable",
                "The watchdog package is not installed, so folders cannot be watched yet.\n\n"
                "Install it with:\n  pip install watchdog\n\n"
                "Then restart TileVision AI.",
            )

    def _on_remove_folder(self) -> None:
        selected = self._folders_list.currentItem()
        if not selected:
            return

        folder_path = selected.text()
        current = [f for f in self._settings.watch_folders if f != folder_path]
        self._settings.watch_folders = current

        self._folders_list.takeItem(self._folders_list.row(selected))
        logger.info(f"Removed watched folder: {folder_path}")
        self._apply_watch_folder_changes()

    # ── Handlers: Preferences ────────────────────────────────────────────

    def _on_top_k_changed(self, value: str) -> None:
        try:
            self._settings.top_k = int(value)
        except ValueError:
            logger.warning(f"Ignoring non-numeric Search Results Shown value: {value!r}")

    def _on_theme_selected(self, theme: str) -> None:
        self._settings.theme = theme
        if self._on_theme_changed:
            self._on_theme_changed(theme)

    # ── Handlers: Maintenance ────────────────────────────────────────────

    def _on_backup_database(self) -> None:
        if self._db_path_provider is None:
            return
        db_path = self._db_path_provider()
        if not db_path.exists():
            QMessageBox.warning(self, "No Database Found", f"Database file not found:\n{db_path}")
            return

        default_name = f"tilevision_backup_{db_path.stem}.db"
        dest_str, _ = QFileDialog.getSaveFileName(
            self, "Backup Database", default_name, "SQLite Database (*.db)"
        )
        if not dest_str:
            return

        try:
            shutil.copy2(db_path, dest_str)
            QMessageBox.information(self, "Backup Complete", f"Database backed up to:\n{dest_str}")
            logger.info(f"Database backed up to {dest_str}")
        except OSError as e:
            QMessageBox.critical(self, "Backup Failed", f"Could not back up database:\n{e}")
            logger.error(f"Database backup failed: {e}")

    def _on_export_logs(self) -> None:
        log_path = get_log_file_path()
        if not log_path.exists():
            QMessageBox.information(self, "No Logs Found", f"No log file found yet at:\n{log_path}")
            return

        dest_str, _ = QFileDialog.getSaveFileName(
            self, "Export Logs", log_path.name, "Log Files (*.log);;All Files (*)"
        )
        if not dest_str:
            return

        try:
            shutil.copy2(log_path, dest_str)
            QMessageBox.information(self, "Logs Exported", f"Logs exported to:\n{dest_str}")
            logger.info(f"Logs exported to {dest_str}")
        except OSError as e:
            QMessageBox.critical(self, "Export Failed", f"Could not export logs:\n{e}")
            logger.error(f"Log export failed: {e}")

    def _on_export_diagnostics(self) -> None:
        dest_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export Diagnostics",
            str(Path.home() / "tilevision_diagnostics.json"),
            "JSON Files (*.json);;All Files (*)",
        )
        if not dest_str:
            return
        try:
            from src.utils.diagnostics import export_diagnostics_json

            info = {}
            if self._diagnostics_info_provider is not None:
                info = dict(self._diagnostics_info_provider() or {})
            path = export_diagnostics_json(dest_str, info)
            QMessageBox.information(
                self,
                "Diagnostics Exported",
                f"Diagnostics report exported to:\n{path}",
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Export Failed", f"Could not export diagnostics:\n{e}"
            )
            logger.error("Diagnostics export failed: %s", e)

    def offer_guided_rebuild(self, summary: str) -> None:
        """Prompt the user to rebuild after an incompatible upgrade."""
        if self._indexing_use_case is None or self._indexed_folders_provider is None:
            QMessageBox.warning(
                self,
                "Rebuild Recommended",
                f"{summary}\n\nOpen Settings when folders are configured, then "
                "use Rebuild FAISS Index.",
            )
            return
        reply = QMessageBox.question(
            self,
            "Guided Rebuild",
            f"{summary}\n\nStart Rebuild FAISS Index now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._on_rebuild_faiss()

    def _on_clear_cache(self) -> None:
        thumb_dir = Path(self._settings.thumbnail_dir)
        if not thumb_dir.exists():
            QMessageBox.information(self, "Nothing to Clear", "No thumbnail cache found.")
            return

        confirm = QMessageBox.question(
            self,
            "Clear Cache",
            "Delete all cached thumbnails? They'll be regenerated automatically the "
            "next time you view search results or the catalog.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        deleted = 0
        errors = 0
        for file_path in thumb_dir.glob("*"):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    deleted += 1
                except OSError:
                    errors += 1

        message = f"Cleared {deleted} cached thumbnail(s)."
        if errors:
            message += f" ({errors} could not be deleted.)"
        QMessageBox.information(self, "Cache Cleared", message)
        logger.info(f"Cleared thumbnail cache: {deleted} deleted, {errors} errors")

    def _on_rebuild_faiss(self) -> None:
        if self._indexing_use_case is None or self._indexed_folders_provider is None:
            return

        folders = self._indexed_folders_provider()
        if not folders:
            QMessageBox.information(self, "Nothing to Rebuild", "No indexed folders found.")
            return

        confirm = QMessageBox.question(
            self,
            "Rebuild Search Index",
            f"This will re-analyze every image in {len(folders)} indexed folder(s) and "
            "rebuild search from scratch. This can take a while for large "
            "catalogs and cannot be cancelled once started.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self._start_rebuild(
            skip_confirm=True,
            window_title="Rebuilding Search Index",
            preparing_text="Preparing rebuild...",
            progress_prefix="Rebuilding Search Index",
        )

    def _start_rebuild(
        self,
        *,
        skip_confirm: bool = False,
        window_title: str = "Rebuilding Search Index",
        preparing_text: str = "Preparing rebuild...",
        progress_prefix: str = "Rebuilding Search Index",
    ) -> None:
        if self._indexing_use_case is None or self._indexed_folders_provider is None:
            return
        folders = self._indexed_folders_provider()
        if not folders:
            return

        # Align in-memory FAISS config with SearchOptimizationEngine decision.
        engine = getattr(self, "_search_optimization_engine", None)
        if engine is not None:
            try:
                catalog_size = 0
                if self._catalog_count_provider is not None:
                    catalog_size = int(self._catalog_count_provider() or 0)
                decision = engine.analyze_and_decide(
                    catalog_size=catalog_size,
                    current_backend=self._settings.index_backend,
                    run_benchmark=False,
                )
                engine.apply_to_settings(self._settings, decision)
                if self._vector_index is not None and hasattr(self._vector_index, "configure_backend"):
                    self._vector_index.configure_backend(decision.selected_backend)
                    # Structure change: clear on-disk index so rebuild embeds into new backend.
                    if decision.rebuild_required and hasattr(self._vector_index, "clear_all"):
                        self._vector_index.clear_all()
            except Exception as exc:
                logger.warning("Pre-rebuild search optimization failed: %s", exc)

        self._rebuild_progress_prefix = progress_prefix
        self._rebuild_button.setEnabled(False)
        self._rebuild_progress_dialog = QProgressDialog(
            preparing_text, None, 0, 1, self
        )
        self._rebuild_progress_dialog.setWindowTitle(window_title)
        self._rebuild_progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._rebuild_progress_dialog.setMinimumDuration(0)
        self._rebuild_progress_dialog.setCancelButton(None)
        self._rebuild_progress_dialog.setAutoClose(False)
        self._rebuild_progress_dialog.setAutoReset(False)
        self._rebuild_progress_dialog.show()

        self._rebuild_worker = RebuildIndexWorker(self._indexing_use_case, folders)
        self._rebuild_worker.progress_updated.connect(self._on_rebuild_progress)
        self._rebuild_worker.rebuild_finished.connect(self._on_rebuild_finished)
        self._rebuild_worker.rebuild_failed.connect(self._on_rebuild_failed)
        self._rebuild_worker.finished.connect(self._rebuild_worker.deleteLater)
        self._rebuild_worker.start()

    def _on_rebuild_progress(
        self,
        processed: int,
        total: int,
        current_name: str,
        eta_seconds: float,
    ) -> None:
        dialog = self._rebuild_progress_dialog
        if dialog is None:
            return

        if total > 0 and dialog.maximum() != total:
            dialog.setMaximum(total)

        dialog.setValue(min(processed, max(total, 1)))
        eta_text = f" — ~{int(eta_seconds)}s remaining" if eta_seconds > 1 else ""
        prefix = getattr(self, "_rebuild_progress_prefix", "Rebuilding Search Index")
        dialog.setLabelText(
            f"{prefix}: {current_name} ({processed}/{total}){eta_text}"
        )

    def _on_rebuild_finished(self, total_reembedded: int, total_failed: int) -> None:
        self._auto_optimize_running = False
        self._rebuild_button.setEnabled(True)
        if self._rebuild_progress_dialog is not None:
            self._rebuild_progress_dialog.setValue(
                self._rebuild_progress_dialog.maximum()
            )
            self._rebuild_progress_dialog.close()
            self._rebuild_progress_dialog = None
        self._rebuild_worker = None

        if self._on_catalog_changed is not None:
            self._on_catalog_changed()
        self.refresh_feature_status()
        self.refresh_search_engine_status(run_analysis=True)

        message = f"Rebuild complete. {total_reembedded} image(s) re-indexed."
        if total_failed:
            message += f" {total_failed} failed."
        QMessageBox.information(self, "Rebuild Complete", message)

    def _on_rebuild_failed(self, error_message: str) -> None:
        self._auto_optimize_running = False
        self._rebuild_button.setEnabled(True)
        if self._rebuild_progress_dialog is not None:
            self._rebuild_progress_dialog.close()
            self._rebuild_progress_dialog = None
        self._rebuild_worker = None
        QMessageBox.critical(self, "Rebuild Failed", error_message)

    def set_theme(self, theme: str) -> None:
        """Re-skin this view for a newly-selected theme (called by MainWindow)."""
        self._theme = theme
        if hasattr(self, "_export_profiles_panel"):
            self._export_profiles_panel.set_theme(theme)
        self._apply_styles()

    def _apply_styles(self) -> None:
        self.setStyleSheet(get_shared_view_qss(self._theme) + get_settings_view_qss(self._theme))
