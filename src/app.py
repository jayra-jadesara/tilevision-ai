"""
Application bootstrapper for TileVision AI.

Responsible for:
    1. Initialising structured logging (earliest possible).
    2. Loading application settings from config.json.
    3. Instantiating and wiring all dependency objects
       (DbContext → Repositories → UseCases → ViewModels → Views).
    4. Performing the offline license gate-check on startup.
    5. Launching the QApplication event loop.

Design Decision:
    All dependency construction is concentrated here (Composition Root / DI Root).
    No other module imports concrete implementations directly — they receive
    interfaces or pre-constructed instances via constructor injection.
    This makes every layer independently unit-testable with mock collaborators.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtGui import QFont, QIcon
from PySide6.QtCore import QTimer

from src.utils.brand_assets import APP_ICON_PATH
from src.utils.logger import setup_logger
from src.config.settings import AppSettings
from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import (
    SQLiteImageRepository, SQLiteLicenseRepository, SQLiteIndexedFolderRepository,
    SQLiteSearchHistoryRepository, SQLiteActivityLogRepository,
)

from src.ai.vector_index import FaissIndexManager
from src.core.use_cases.index_images import IndexImagesUseCase
from src.core.use_cases.search_tiles import SearchTilesUseCase
from src.core.use_cases.monitor_folder import FolderMonitorController, is_watchdog_available
from src.core.use_cases.find_duplicates import FindDuplicatesUseCase
from src.core.use_cases.validate_license import ValidateLicenseUseCase
from src.licensing.trial_manager import TrialManager
from src.licensing.validator import LicenseValidator
from src.presentation.viewmodels.indexing_viewmodel import IndexingViewModel
from src.presentation.viewmodels.search_viewmodel import SearchViewModel
from src.presentation.views.main_window import MainWindow, DashboardDataProviders
from src.presentation.views.license_view import LicenseView
from src.presentation.auto_index_notifier import AutoIndexNotifier
from src.core.use_cases.monitor_folder import AutoIndexAction

_app_logger = logging.getLogger("tilevision.app")


def _on_auto_indexed(
    file_path: str,
    action: AutoIndexAction,
    success: bool,
    error_message: str,
    *,
    activity_log_repository,
    auto_index_notifier: AutoIndexNotifier,
) -> None:
    """
    Callback invoked by FolderMonitorController after auto-index events.
    Runs on the watchdog background thread — must not touch QWidget directly.
    """
    name = Path(file_path).name
    if action == "indexed" and success:
        _app_logger.info("Auto-indexed file: %s", file_path)
        activity_log_repository.record_activity("auto_index", f"Auto-indexed: {name}")
    elif action == "removed" and success:
        _app_logger.info("Auto-removed deleted file from index: %s", file_path)
        activity_log_repository.record_activity("auto_index", f"Removed from index: {name}")
    elif action == "failed":
        _app_logger.warning("Auto-indexing failed for %s: %s", file_path, error_message)
    elif action == "skipped":
        _app_logger.debug("Auto-monitor skipped unchanged file: %s", file_path)
        return

    # Marshal UI refresh onto the Qt main thread.
    QTimer.singleShot(
        0,
        lambda: auto_index_notifier.notify(file_path, action, success),
    )


def build_application() -> int:
    """
    Compose and launch the full TileVision AI application.

    This is the single entry point that wires everything together.

    Returns:
        The QApplication exit code (0 for clean exit).
    """
    # ── 1. Configure Logging first (before any other module runs) ────────────
    root_logger = setup_logger(
        name="tilevision",
        log_file_name="tilevision.log",
        log_level=logging.INFO,
    )
    logger = logging.getLogger("tilevision.app")
    logger.info("═" * 60)
    logger.info("TileVision AI — Starting application")
    logger.info("═" * 60)

    # ── 2. Load Settings ──────────────────────────────────────────────────────
    settings = AppSettings()
    logger.info(f"Configuration loaded. Data directory: {Path(settings.database_path).parent}")

    # ── 3. Create QApplication (must happen before any QWidget is created) ────
    app = QApplication(sys.argv)
    app.setApplicationName("TileVision AI")
    app.setOrganizationName("JD Software")
    app.setApplicationVersion("1.0.0")

    if APP_ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(APP_ICON_PATH)))

    # Set modern default font
    default_font = QFont("Segoe UI", 10)
    app.setFont(default_font)

    # ── 3b. First-run dependency setup wizard ─────────────────────────────────
    from src.presentation.views.setup_wizard import SetupWizardDialog, should_show_setup_wizard

    if should_show_setup_wizard(settings):
        logger.info("Showing first-run setup wizard...")
        wizard = SetupWizardDialog(settings, theme=settings.theme)
        if wizard.exec() != SetupWizardDialog.DialogCode.Accepted:
            logger.warning("Setup wizard cancelled — exiting.")
            return 1

    # ── 4. Construct Data Layer ───────────────────────────────────────────────
    logger.info("Initializing database context...")
    db_context = DatabaseContext(db_path=settings.database_path)

    image_repository = SQLiteImageRepository(db_context=db_context)
    license_repository = SQLiteLicenseRepository(db_context=db_context)
    indexed_folder_repository = SQLiteIndexedFolderRepository(db_context=db_context)
    search_history_repository = SQLiteSearchHistoryRepository(db_context=db_context)
    activity_log_repository = SQLiteActivityLogRepository(db_context=db_context)

    # ── 5. Construct Licensing Layer ──────────────────────────────────────────
    logger.info("Initializing license validator...")
    license_validator = LicenseValidator()
    trial_manager = TrialManager()
    validate_license_use_case = ValidateLicenseUseCase(
        license_repository=license_repository,
        validator=license_validator,
        trial_manager=trial_manager,
    )

    # ── 6. License Gate on Startup ────────────────────────────────────────────
    # verify_existing_license() returns a non-None result during an ACTIVE
    # trial too (not just a paid license), so the blocking activation
    # dialog below only appears when there's truly no paid license AND no
    # active/valid trial remaining.
    logger.info("Checking startup license status...")
    license_details = validate_license_use_case.verify_existing_license()

    if license_details is None:
        logger.warning("No valid license or active trial found — showing activation dialog.")
        license_dialog = LicenseView(validate_use_case=validate_license_use_case)
        result = license_dialog.exec()

        if not license_dialog.is_activated:
            logger.warning("License activation skipped or failed. Exiting.")
            QMessageBox.critical(
                None,
                "License Required",
                "TileVision AI requires a valid license or trial to run.\n\n"
                "Please contact your supplier for a license key.\n\n"
                "The application will now close.",
            )
            return 1

        # Re-fetch so the main window has fresh (post-activation) details.
        license_details = validate_license_use_case.verify_existing_license()
    else:
        customer = license_details.get("customer_name", "Unknown")
        if license_details.get("is_trial"):
            logger.info(
                f"Active trial: {license_details.get('days_remaining')} day(s) remaining."
            )
        else:
            logger.info(f"Valid license found for: {customer}")

    # ── 7. Construct AI Layer ─────────────────────────────────────────────────
    from src.ai.embedder import DINOv2Embedder
    from src.ai.feature_extractor import FeatureExtractor
    from src.ai.preprocess.image_preprocessor import ImagePreprocessor
    from src.config.indexing_performance import IndexingPerformanceConfig

    logger.info("Initializing AI engine...")
    embedder = DINOv2Embedder(device_preference=settings.inference_device)
    indexing_perf = IndexingPerformanceConfig.from_settings(
        settings,
        use_gpu=embedder.using_gpu,
    )
    ImagePreprocessor.configure(max_decode_edge=indexing_perf.max_decode_edge)
    logger.info(
        "Indexing performance: device=%s batch=%d checkpoint=%d max_decode=%d workers=%d",
        embedder.runtime_info.summary_for_ui(),
        indexing_perf.batch_size,
        indexing_perf.checkpoint_interval,
        indexing_perf.max_decode_edge,
        indexing_perf.preprocess_workers,
    )

    feature_extractor = FeatureExtractor(
        embedder=embedder,
        preprocess_workers=indexing_perf.preprocess_workers,
    )

    vector_index = FaissIndexManager(
        index_path=settings.index_path,
        dimension=1024,
    )

    # ── 8. Construct Use Cases ────────────────────────────────────────────────
    logger.info("Initializing use cases...")
    index_images_use_case = IndexImagesUseCase(
        image_repository=image_repository,
        feature_extractor=feature_extractor,
        vector_index=vector_index,
        thumbnail_dir=settings.thumbnail_dir,
        folder_repository=indexed_folder_repository,
        performance=indexing_perf,
    )
    search_tiles_use_case = SearchTilesUseCase(
        image_repository=image_repository,
        feature_extractor=feature_extractor,
        vector_index=vector_index,
        thumbnail_dir=settings.thumbnail_dir,
    )
    find_duplicates_use_case = FindDuplicatesUseCase(image_repository=image_repository, vector_index=vector_index)

    # ── 8b. Warm up the CLIP model and FAISS index now, synchronously, so the
    #        *first* search a user runs is fast rather than paying model-load
    #        cost on that click. Loading is a one-time, ~1-3s startup cost;
    #        after this, both index_images and search_tiles reuse the same
    #        in-memory model/index instances for the lifetime of the process.
    try:
        logger.info("Warming up AI engine and FAISS index...")
        feature_extractor.load_model()
        vector_index.load_index()
        version_status = image_repository.get_feature_version_status()
        if not version_status.is_compatible and version_status.stale_count > 0:
            logger.warning(
                "Stale feature index detected: %s "
                "Use Settings > Rebuild FAISS Index after re-scanning folders.",
                version_status.message,
            )
        logger.info("AI engine warm-up complete.")
    except Exception as e:
        # Non-fatal: indexing/search will lazily retry loading on first use
        # and surface a clear error there if the AI engine truly can't load.
        logger.error(f"AI engine warm-up failed (will retry on first use): {e}")

    # ── 8c. Start Auto Folder Monitoring (Feature 7) ──────────────────────────
    # Watches settings.watch_folders in the background (watchdog) and
    # automatically indexes any new/changed image dropped into them,
    # without the user needing to run a manual folder scan. Uses the same
    # index_images_use_case instance as manual indexing — new files still
    # go through the identical embed → FAISS → SQLite pipeline.
    folder_monitor: Optional[FolderMonitorController] = None
    auto_index_notifier = AutoIndexNotifier()
    watch_folders = settings.watch_folders

    def _auto_index_callback(path: str, action: AutoIndexAction, success: bool, message: str) -> None:
        _on_auto_indexed(
            path,
            action,
            success,
            message,
            activity_log_repository=activity_log_repository,
            auto_index_notifier=auto_index_notifier,
        )

    def _create_folder_monitor() -> Optional[FolderMonitorController]:
        nonlocal folder_monitor
        if not is_watchdog_available():
            return None
        try:
            folder_monitor = FolderMonitorController(
                indexing_use_case=index_images_use_case,
                on_file_indexed_callback=_auto_index_callback,
            )
            return folder_monitor
        except Exception as exc:
            logger.error("Failed to create folder monitor: %s", exc)
            folder_monitor = None
            return None

    def _restart_folder_monitor() -> None:
        nonlocal folder_monitor
        folders = settings.watch_folders
        if not folders:
            if folder_monitor is not None:
                folder_monitor.stop_monitoring()
            logger.info("Auto folder monitoring stopped (no watched folders).")
            return

        if folder_monitor is None:
            if not _create_folder_monitor():
                logger.error(
                    "Cannot monitor folders: watchdog is not installed. "
                    "Run: pip install watchdog — then restart the app."
                )
                return

        try:
            folder_monitor.restart_monitoring(folders)
            logger.info(
                "Restarted auto folder monitoring for %d folder(s).",
                len(folders),
            )
        except Exception as exc:
            logger.error("Failed to restart folder monitoring: %s", exc)

    if watch_folders:
        try:
            logger.info(f"Starting auto folder monitoring for {len(watch_folders)} folder(s)...")
            if _create_folder_monitor() is not None:
                folder_monitor.start_monitoring(watch_folders)
            else:
                raise ImportError("watchdog package is required for FolderMonitorController.")
        except Exception as e:
            logger.error(f"Failed to start folder monitoring (continuing without it): {e}")
            folder_monitor = None
    elif not is_watchdog_available():
        logger.warning(
            "watchdog is not installed — auto folder monitoring will not work until "
            "you run: pip install watchdog"
        )
    else:
        logger.info("No watched folders configured — auto folder monitoring not started.")

    # ── 9. Construct ViewModels ───────────────────────────────────────────────
    logger.info("Constructing view models...")
    indexing_viewmodel = IndexingViewModel(
        use_case=index_images_use_case, activity_log_repository=activity_log_repository
    )
    search_viewmodel = SearchViewModel(
        use_case=search_tiles_use_case,
        default_top_k=settings.top_k,
        search_history_repository=search_history_repository,
        activity_log_repository=activity_log_repository,
    )

    # ── 10. Launch Main Window ────────────────────────────────────────────────
    logger.info("Launching main application window...")

    def _get_file_size(path) -> int:
        try:
            return path.stat().st_size if path.exists() else 0
        except OSError:
            return 0

    dashboard_providers = DashboardDataProviders(
        indexed_folder_count=lambda: len(indexed_folder_repository.get_all_folders()),
        database_size=lambda: _get_file_size(db_context.db_path),
        faiss_size=lambda: _get_file_size(vector_index.index_path),
        last_search=search_history_repository.get_last_search,
        recent_activity=lambda: activity_log_repository.get_recent_activity(limit=8),
        recent_searches=lambda: search_history_repository.get_recent_searches(limit=8),
    )

    main_window = MainWindow(
        indexing_viewmodel=indexing_viewmodel,
        search_viewmodel=search_viewmodel,
        license_details=license_details,
        find_duplicates_use_case=find_duplicates_use_case,
        settings=settings,
        catalog_count_provider=lambda: len(image_repository.get_all()),
        dashboard_providers=dashboard_providers,
        db_path_provider=lambda: db_context.db_path,
        indexing_use_case=index_images_use_case,
        indexed_folders_provider=lambda: [f.folder_path for f in indexed_folder_repository.get_all_folders()],
        feature_version_provider=image_repository.get_feature_version_status,
        gpu_info_provider=lambda: embedder.runtime_info,
        on_watch_folders_changed=_restart_folder_monitor,
    )
    auto_index_notifier.catalog_updated.connect(main_window.handle_auto_index_event)
    main_window.show()

    logger.info("TileVision AI is running.")

    # ── 11. Run Qt Event Loop ─────────────────────────────────────────────────
    exit_code = app.exec()

    if folder_monitor is not None:
        logger.info("Stopping folder monitor before shutdown...")
        folder_monitor.stop_monitoring()

    logger.info(f"TileVision AI exiting with code: {exit_code}")
    return exit_code
