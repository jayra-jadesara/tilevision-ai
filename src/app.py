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

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QIcon
from PySide6.QtCore import QTimer


from src.version import APP_VERSION
from src.utils.logger import setup_logger
from src.config.settings import AppSettings
from src.data.db_context import DatabaseContext
from src.data.db_protection import seal_database
from src.data.sqlite_repository import (
    SQLiteImageRepository, SQLiteLicenseRepository, SQLiteIndexedFolderRepository,
    SQLiteSearchHistoryRepository, SQLiteActivityLogRepository,
    SQLiteCatalogueProfileRepository,
)

from src.ai.vector_index import FaissIndexManager
from src.core.use_cases.index_images import IndexImagesUseCase
from src.core.use_cases.search_tiles import SearchTilesUseCase
from src.core.use_cases.monitor_folder import FolderMonitorController, is_watchdog_available
from src.core.use_cases.find_duplicates import FindDuplicatesUseCase
from src.core.use_cases.validate_license import ValidateLicenseUseCase
from src.theme.theme_manager import get_app_stylesheet
from src.licensing.validator import LicenseValidator
from src.presentation.viewmodels.indexing_viewmodel import IndexingViewModel
from src.presentation.viewmodels.search_viewmodel import SearchViewModel
from src.presentation.views.main_window import MainWindow, DashboardDataProviders
from src.presentation.views.license_view import LicenseView
from src.presentation.auto_index_notifier import AutoIndexNotifier
from src.presentation.update_controller import UpdateController
from src.utils.platform_info import app_icon_path, default_ui_font_family
from src.core.use_cases.monitor_folder import AutoIndexAction
from src.presentation.dialogs import message_box

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
        log_level=None,  # Release → WARNING console; see resolve_log_level()
    )
    logger = logging.getLogger("tilevision.app")
    logger.info("═" * 60)
    logger.info("TileVision AI — Starting application")
    logger.info("═" * 60)

    # ── 2. Load Settings ──────────────────────────────────────────────────────
    settings = AppSettings()
    logger.info(f"Configuration loaded. Data directory: {Path(settings.database_path).parent}")

    from src.ai.preprocess.sam2_backend import configure_sam2_from_settings

    configure_sam2_from_settings(settings.enable_sam2_precise_crop)

    # ── 3. Create QApplication (must happen before any QWidget is created) ────
    app = QApplication(sys.argv)
    app.setApplicationName("TileVision AI")
    app.setOrganizationName("JD Software")
    app.setApplicationVersion(APP_VERSION)

    icon_path = app_icon_path()
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))

    default_font = QFont(default_ui_font_family(), 10)
    app.setFont(default_font)
    app.setStyleSheet(get_app_stylesheet(settings.theme))
    app.setProperty("tilevision_theme", settings.theme)

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
    catalogue_profile_repository = SQLiteCatalogueProfileRepository(db_context=db_context)

    # ── 5. Construct Licensing Layer ──────────────────────────────────────────
    logger.info("Initializing license validator...")
    license_validator = LicenseValidator()
    validate_license_use_case = ValidateLicenseUseCase(
        license_repository=license_repository,
        validator=license_validator,
    )

    # ── 6. License Gate on Startup ────────────────────────────────────────────
    logger.info("Checking startup license status...")
    license_details = validate_license_use_case.verify_existing_license()

    if license_details is None:
        logger.info("Showing license activation dialog.")
        license_dialog = LicenseView(
            validate_use_case=validate_license_use_case,
            theme=settings.theme,
            show_back=True,
        )
        license_dialog.exec()

        if not license_dialog.is_activated:
            logger.warning("License activation skipped or failed. Exiting.")
            message_box.critical(
                None,
                "License Required",
                "TileVision AI requires a valid license key to run.\n\n"
                "Please contact your supplier for a trial or full license key.\n\n"
                "The application will now close.",
            )
            return 1

        license_details = validate_license_use_case.verify_existing_license()
    else:
        customer = license_details.get("customer_name", "Unknown")
        if license_details.get("is_trial"):
            logger.info(
                f"Active trial license: {license_details.get('days_remaining')} day(s) remaining."
            )
        else:
            logger.info(f"Valid license found for: {customer}")

    customer_name = str((license_details or {}).get("customer_name") or "").strip()
    from src.services.catalogue_master_service import CatalogueMasterService

    catalogue_master_service = CatalogueMasterService(
        repository=catalogue_profile_repository,
        license_customer_name=customer_name,
    )
    if customer_name:
        catalogue_master_service.migrate_legacy_storage_if_needed()
        catalogue_master_service.ensure_profile_for_customer(customer_name)

    # ── 7. Construct AI Layer ─────────────────────────────────────────────────
    from src.ai.embedder import DINOv2Embedder
    from src.ai.feature_extractor import FeatureExtractor
    from src.ai.gpu_info import configure_mps_fallback
    from src.ai.preprocess.image_preprocessor import ImagePreprocessor
    from src.config.indexing_performance import IndexingPerformanceConfig

    # Mac: must run before DINOv2/MPS inference (missing Metal ops → CPU).
    configure_mps_fallback()

    # macOS Intel: OpenMP inside Qt QThread deadlocks DINOv2/FAISS search.
    # Remap AI workers onto Python threads and cap OpenMP before model load.
    from src.presentation.workers.native_ai_thread import (
        apply_torch_faiss_thread_caps,
        install_python_ai_worker_threads,
    )

    install_python_ai_worker_threads()
    apply_torch_faiss_thread_caps()

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

    from src.ai.index_backends import BackendParams, IndexBackend

    backend = IndexBackend.parse(settings.index_backend)
    backend_params = BackendParams(
        hnsw_m=settings.hnsw_m,
        hnsw_ef_search=settings.hnsw_ef_search,
        ivf_nlist=settings.ivf_nlist,
        ivf_nprobe=settings.ivf_nprobe,
        ivf_pq_m=settings.ivf_pq_m,
    )
    vector_index = FaissIndexManager(
        index_path=settings.index_path,
        dimension=1024,
        backend=backend,
        backend_params=backend_params,
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
    import time as _time

    from src.ai.feature_versions import CURRENT_EMBEDDING_DIMENSION, CURRENT_EMBEDDING_MODEL
    from src.core.compatibility import run_compatibility_check
    from src.utils.diagnostics import log_startup_diagnostics
    from src.utils.pipeline_timing import profiling_enabled

    compatibility_report = None
    try:
        logger.info("Warming up AI engine and FAISS index...")
        t0 = _time.perf_counter()
        feature_extractor.load_model()
        model_ms = (_time.perf_counter() - t0) * 1000.0
        t1 = _time.perf_counter()
        vector_index.load_index()
        faiss_ms = (_time.perf_counter() - t1) * 1000.0
        version_status = image_repository.get_feature_version_status()
        if not version_status.is_compatible and version_status.stale_count > 0:
            logger.warning(
                "Stale feature index detected: %s "
                "Use Settings > Rebuild Search Index after re-scanning folders.",
                version_status.message,
            )
        compatibility_report = run_compatibility_check(
            database_path=settings.database_path,
            index_path=settings.index_path,
            expected_backend=backend,
            feature_status_provider=image_repository.get_feature_version_status,
            catalog_size=vector_index.get_total_count(),
        )
        logger.info(
            "AI engine warm-up complete (model=%.0f ms, faiss=%.0f ms).",
            model_ms,
            faiss_ms,
        )
        try:
            import torch as _torch

            torch_ver = getattr(_torch, "__version__", "unknown")
        except Exception:
            torch_ver = "unavailable"
        try:
            import faiss as _faiss

            faiss_ver = getattr(_faiss, "__version__", "installed")
            omp = int(_faiss.omp_get_max_threads())
        except Exception:
            faiss_ver = "unavailable"
            omp = 0
        log_startup_diagnostics(
            {
                "app_version": APP_VERSION,
                "python": sys.version.split()[0],
                "torch": torch_ver,
                "faiss": faiss_ver,
                "embedding_model": CURRENT_EMBEDDING_MODEL,
                "embedding_dim": CURRENT_EMBEDDING_DIMENSION,
                "device": embedder.runtime_info.summary_for_ui(),
                "faiss_type": vector_index.index_type_name(),
                "index_backend": vector_index.active_backend().value,
                "omp_threads": omp,
                "catalog_size": vector_index.get_total_count(),
                "database": settings.database_path,
                "index_path": settings.index_path,
                "profile_enabled": profiling_enabled(),
                "log_level": logging.getLevelName(root_logger.level),
                "model_warmup_ms": round(model_ms, 1),
                "faiss_warmup_ms": round(faiss_ms, 1),
                "compatibility": compatibility_report.to_dict(),
                "gpu": embedder.runtime_info.device_name or embedder.runtime_info.active_device,
            }
        )
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

    def _on_search_busy_changed(busy: bool) -> None:
        # Drop-image search takes priority over background indexing.
        if busy:
            indexing_viewmodel.pause_for_search()
            if folder_monitor is not None:
                folder_monitor.pause_for_search()
        else:
            indexing_viewmodel.resume_after_search()
            if folder_monitor is not None:
                folder_monitor.resume_after_search()

    search_viewmodel = SearchViewModel(
        use_case=search_tiles_use_case,
        default_top_k=settings.top_k,
        search_history_repository=search_history_repository,
        activity_log_repository=activity_log_repository,
        on_search_busy_changed=_on_search_busy_changed,
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

    update_controller = UpdateController(settings, theme=settings.theme)

    def _diagnostics_info() -> dict:
        from src.ai.feature_versions import (
            CURRENT_EMBEDDING_DIMENSION,
            CURRENT_EMBEDDING_MODEL,
        )
        from src.utils.pipeline_timing import profiling_enabled

        try:
            import faiss as _faiss

            faiss_ver = getattr(_faiss, "__version__", "installed")
            omp = int(_faiss.omp_get_max_threads())
        except Exception:
            faiss_ver = "unavailable"
            omp = 0
        try:
            import torch as _torch

            torch_ver = getattr(_torch, "__version__", "unknown")
        except Exception:
            torch_ver = "unavailable"
        payload = {
            "app_version": APP_VERSION,
            "python": sys.version.split()[0],
            "torch": torch_ver,
            "faiss": faiss_ver,
            "embedding_model": CURRENT_EMBEDDING_MODEL,
            "embedding_dim": CURRENT_EMBEDDING_DIMENSION,
            "device": embedder.runtime_info.summary_for_ui(),
            "gpu": embedder.runtime_info.device_name or embedder.runtime_info.active_device,
            "faiss_type": vector_index.index_type_name(),
            "index_backend": vector_index.active_backend().value,
            "omp_threads": omp,
            "catalog_size": vector_index.get_total_count(),
            "database": settings.database_path,
            "index_path": settings.index_path,
            "profile_enabled": profiling_enabled(),
            "log_level": logging.getLevelName(root_logger.level),
        }
        if compatibility_report is not None:
            payload["compatibility"] = compatibility_report.to_dict()
        return payload

    main_window = MainWindow(
        indexing_viewmodel=indexing_viewmodel,
        search_viewmodel=search_viewmodel,
        license_details=license_details,
        catalogue_master_service=catalogue_master_service,
        find_duplicates_use_case=find_duplicates_use_case,
        settings=settings,
        catalog_count_provider=lambda: len(image_repository.get_all()),
        dashboard_providers=dashboard_providers,
        db_path_provider=lambda: db_context.db_path,
        indexing_use_case=index_images_use_case,
        indexed_folders_provider=lambda: [f.folder_path for f in indexed_folder_repository.get_all_folders()],
        feature_version_provider=image_repository.get_feature_version_status,
        gpu_info_provider=lambda: embedder.runtime_info,
        diagnostics_info_provider=_diagnostics_info,
        on_watch_folders_changed=_restart_folder_monitor,
        on_check_updates=lambda: update_controller.check_now(main_window),
        vector_index=vector_index,
    )
    auto_index_notifier.catalog_updated.connect(main_window.handle_auto_index_event)
    main_window.show()
    update_controller.schedule_startup_check(main_window)

    if compatibility_report is not None and compatibility_report.requires_rebuild:
        summary = compatibility_report.summary_message()

        def _offer_rebuild() -> None:
            main_window.offer_compatibility_rebuild(summary)

        QTimer.singleShot(750, _offer_rebuild)

    logger.info("TileVision AI is running.")

    # ── 11. Run Qt Event Loop ─────────────────────────────────────────────────
    exit_code = app.exec()

    if folder_monitor is not None:
        logger.info("Stopping folder monitor before shutdown...")
        folder_monitor.stop_monitoring()

    from src.utils.update_installer import is_force_quit_for_update

    updating = False
    try:
        updating = is_force_quit_for_update()
    except Exception:
        logger.exception("Could not read update force-quit flag")

    # During in-app update we must not seal (encrypt) the DB: a hard os._exit
    # can interrupt AES write and leave a corrupt .enc that crashes next launch.
    # AppData is not replaced by Inno; plaintext working DB is fine across upgrade.
    if not updating:
        try:
            seal_database(db_context.db_path)
            logger.info("Catalogue database encrypted at rest.")
        except Exception as exc:
            logger.error("Failed to encrypt catalogue database on exit: %s", exc)
    else:
        logger.info("Skipping DB seal — quitting for in-app update installer.")

    logger.info(f"TileVision AI exiting with code: {exit_code}")

    if updating:
        logger.info("Hard-exiting so the update installer can replace this build.")
        import os

        os._exit(0)

    return exit_code
