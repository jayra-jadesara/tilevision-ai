"""
Settings View for TileVision AI (Task D: Settings).

Lets the user configure:
  - Theme (dark/light), thumbnail size, number of search results.
  - Watched folders for auto-indexing (Feature 7) — add/remove.
  - Language (placeholder — English only for now).
  - Backup Database (uses the database path internally; not shown to the
    user — an absolute file path invites accidental navigation/deletion).
  - Rebuild Search Index (force re-embed everything).
  - Clear thumbnail Cache.
  - Export Logs.
"""

import logging
import os
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
from src.presentation.dialogs import message_box

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
            message_box.warning(self, "Unavailable", "Machine ID could not be read on this PC.")
            return
        QGuiApplication.clipboard().setText(machine_id)
        message_box.information(
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
        """Customer Search Engine health — no FAISS/backend jargon."""
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

        # Developer Mode only (TILEVISION_DEV_MODE=1). Uses index_advisor internally.
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
        self._dev_refresh_advisor_button = QPushButton("Refresh Advisor")
        self._dev_refresh_advisor_button.setObjectName("SecondaryButton")
        self._dev_refresh_advisor_button.clicked.connect(self._refresh_developer_search_diagnostics)
        dev_layout.addWidget(self._dev_refresh_advisor_button)
        layout.addWidget(self._dev_search_box)
        self._dev_search_box.setVisible(os.environ.get("TILEVISION_DEV_MODE") == "1")

        self.refresh_search_engine_status()
        return box

    def refresh_search_engine_status(self, *, run_analysis: bool = False) -> None:
        """Refresh customer Search Engine health from live catalog data."""
        catalog_size = 0
        try:
            if self._catalog_count_provider is not None:
                catalog_size = int(self._catalog_count_provider() or 0)
        except Exception as exc:
            logger.warning("Could not read catalog size for Search Engine: %s", exc)

        index_status = "Healthy"
        status_line = "✓ Ready"
        if self._feature_version_provider is not None:
            try:
                feat = self._feature_version_provider()
                if feat.indexed_count == 0:
                    status_line = "Ready"
                    index_status = "No catalog yet"
                elif not feat.is_compatible and feat.stale_count > 0:
                    status_line = "Update recommended"
                    index_status = "Needs rebuild"
            except Exception as exc:
                logger.warning("Feature status for Search Engine failed: %s", exc)

        last_indexed = "—"
        try:
            if self._indexing_use_case is not None:
                getter = getattr(self._indexing_use_case, "get_last_indexed_folder_status", None)
                if getter is not None:
                    st = getter()
                    when = getattr(st, "last_indexed_at", None) if st is not None else None
                    if when is not None:
                        last_indexed = when.strftime("%Y-%m-%d %H:%M") if hasattr(when, "strftime") else str(when)
        except Exception as exc:
            logger.debug("Last indexed lookup failed: %s", exc)

        catalog_label = f"{catalog_size:,} Tile" if catalog_size == 1 else f"{catalog_size:,} Tiles"
        self._search_engine_status_label.setText(
            f"Status\n{status_line}\n\n"
            f"Search Quality\nExact\n\n"
            f"Catalog\n{catalog_label}\n\n"
            f"Index Status\n{index_status}\n\n"
            f"Last Indexed\n{last_indexed}"
        )

        if getattr(self, "_dev_search_box", None) is not None and self._dev_search_box.isVisible():
            self._refresh_developer_search_diagnostics()

    def _refresh_developer_search_diagnostics(self) -> None:
        """Internal index_advisor diagnostics — Developer Mode only."""
        if os.environ.get("TILEVISION_DEV_MODE") != "1":
            return
        if not hasattr(self, "_dev_search_diag_label"):
            return
        try:
            from src.ai.feature_versions import CURRENT_EMBEDDING_DIMENSION
            from src.ai.index_advisor import advise_index_backend
            from src.ai.index_backends import BackendParams

            catalog_size = 0
            if self._catalog_count_provider is not None:
                catalog_size = int(self._catalog_count_provider() or 0)
            advice = advise_index_backend(
                catalog_size=catalog_size,
                current_backend="flat_ip",
                embedding_dimension=CURRENT_EMBEDDING_DIMENSION,
                backend_params=BackendParams(
                    hnsw_m=self._settings.hnsw_m,
                    hnsw_ef_search=self._settings.hnsw_ef_search,
                    ivf_nlist=self._settings.ivf_nlist,
                    ivf_nprobe=self._settings.ivf_nprobe,
                    ivf_pq_m=self._settings.ivf_pq_m,
                ),
            )
            active = "flat_ip"
            vector_index = getattr(self, "_vector_index", None)
            if vector_index is not None:
                try:
                    active = vector_index.active_backend().value
                except Exception:
                    try:
                        active = vector_index.configured_backend.value
                    except Exception:
                        active = "flat_ip"
            rows = "\n".join(
                f"- {r.backend}: exact={r.exact} recall={r.recall} mem={r.memory} speed={r.speed}"
                for r in advice.comparison
            )
            self._dev_search_diag_label.setText(
                f"Production policy: flat_ip (exact)\n"
                f"Active backend: {active}\n"
                f"Advisor recommended: {advice.recommended_backend.value}\n"
                f"Reason: {advice.reason}\n"
                f"Expected recall: {advice.expected_recall:.2f}\n"
                f"Expected search: ~{advice.expected_search_ms:.1f} ms\n"
                f"Estimated RAM: ~{advice.estimated_ram_mib:.0f} MiB\n"
                f"Catalog: {advice.catalog_size:,}  dim={advice.embedding_dimension}\n"
                f"Comparison:\n{rows}\n"
                f"{advice.approximate_warning}"
            )
        except Exception as exc:
            self._dev_search_diag_label.setText(f"Advisor unavailable: {exc}")

    def refresh_index_advisor(self) -> None:
        """Compatibility alias for older callers."""
        self.refresh_search_engine_status()

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
            "Rebuild Search Index re-analyzes every tile after a software update. "
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

        self._rebuild_button = QPushButton("Rebuild Search Index")
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
            message_box.information(self, "Already Added", "This folder is already being watched.")
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
            message_box.warning(
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
            message_box.warning(self, "No Database Found", f"Database file not found:\n{db_path}")
            return

        default_name = f"tilevision_backup_{db_path.stem}.db"
        dest_str, _ = QFileDialog.getSaveFileName(
            self, "Backup Database", default_name, "SQLite Database (*.db)"
        )
        if not dest_str:
            return

        try:
            shutil.copy2(db_path, dest_str)
            message_box.information(self, "Backup Complete", f"Database backed up to:\n{dest_str}")
            logger.info(f"Database backed up to {dest_str}")
        except OSError as e:
            message_box.critical(self, "Backup Failed", f"Could not back up database:\n{e}")
            logger.error(f"Database backup failed: {e}")

    def _on_export_logs(self) -> None:
        log_path = get_log_file_path()
        if not log_path.exists():
            message_box.information(self, "No Logs Found", f"No log file found yet at:\n{log_path}")
            return

        dest_str, _ = QFileDialog.getSaveFileName(
            self, "Export Logs", log_path.name, "Log Files (*.log);;All Files (*)"
        )
        if not dest_str:
            return

        try:
            shutil.copy2(log_path, dest_str)
            message_box.information(self, "Logs Exported", f"Logs exported to:\n{dest_str}")
            logger.info(f"Logs exported to {dest_str}")
        except OSError as e:
            message_box.critical(self, "Export Failed", f"Could not export logs:\n{e}")
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
            message_box.information(
                self,
                "Diagnostics Exported",
                f"Diagnostics report exported to:\n{path}",
            )
        except Exception as e:
            message_box.critical(
                self, "Export Failed", f"Could not export diagnostics:\n{e}"
            )
            logger.error("Diagnostics export failed: %s", e)

    def offer_guided_rebuild(self, summary: str) -> None:
        """Prompt the user to rebuild after an incompatible upgrade."""
        if self._indexing_use_case is None or self._indexed_folders_provider is None:
            message_box.warning(
                self,
                "Rebuild Recommended",
                f"{summary}\n\nOpen Settings when folders are configured, then "
                "use Rebuild Search Index.",
            )
            return
        reply = message_box.question(
            self,
            "Guided Rebuild",
            f"{summary}\n\nStart Rebuild Search Index now?",
            message_box.StandardButton.Yes | message_box.StandardButton.No,
            message_box.StandardButton.Yes,
        )
        if reply == message_box.StandardButton.Yes:
            self._on_rebuild_faiss()

    def _on_clear_cache(self) -> None:
        thumb_dir = Path(self._settings.thumbnail_dir)
        if not thumb_dir.exists():
            message_box.information(self, "Nothing to Clear", "No thumbnail cache found.")
            return

        confirm = message_box.question(
            self,
            "Clear Cache",
            "Delete all cached thumbnails? They'll be regenerated automatically the "
            "next time you view search results or the catalog.",
            message_box.StandardButton.Yes | message_box.StandardButton.No,
            message_box.StandardButton.No,
        )
        if confirm != message_box.StandardButton.Yes:
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
        message_box.information(self, "Cache Cleared", message)
        logger.info(f"Cleared thumbnail cache: {deleted} deleted, {errors} errors")

    def _on_rebuild_faiss(self) -> None:
        if self._indexing_use_case is None or self._indexed_folders_provider is None:
            return

        folders = self._indexed_folders_provider()
        if not folders:
            message_box.information(self, "Nothing to Rebuild", "No indexed folders found.")
            return

        confirm = message_box.question(
            self,
            "Rebuild Search Index",
            f"This will re-analyze every image in {len(folders)} indexed folder(s) and "
            "rebuild search from scratch. This can take a while for large "
            "catalogs and cannot be cancelled once started.\n\nContinue?",
            message_box.StandardButton.Yes | message_box.StandardButton.No,
            message_box.StandardButton.No,
        )
        if confirm != message_box.StandardButton.Yes:
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

        # Ensure production FlatIP configuration before rebuild.
        self._settings.index_backend = "flat_ip"
        if self._vector_index is not None and hasattr(self._vector_index, "configure_backend"):
            try:
                self._vector_index.configure_backend("flat_ip")
            except Exception as exc:
                logger.warning("Could not configure FlatIP before rebuild: %s", exc)

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
        self.refresh_search_engine_status()

        message = f"Rebuild complete. {total_reembedded} image(s) re-indexed."
        if total_failed:
            message += f" {total_failed} failed."
        message_box.information(self, "Rebuild Complete", message)

    def _on_rebuild_failed(self, error_message: str) -> None:
        self._rebuild_button.setEnabled(True)
        if self._rebuild_progress_dialog is not None:
            self._rebuild_progress_dialog.close()
            self._rebuild_progress_dialog = None
        self._rebuild_worker = None
        message_box.critical(self, "Rebuild Failed", error_message)

    def set_theme(self, theme: str) -> None:
        """Re-skin this view for a newly-selected theme (called by MainWindow)."""
        self._theme = theme
        if hasattr(self, "_export_profiles_panel"):
            self._export_profiles_panel.set_theme(theme)
        self._apply_styles()

    def _apply_styles(self) -> None:
        self.setStyleSheet(get_shared_view_qss(self._theme) + get_settings_view_qss(self._theme))
