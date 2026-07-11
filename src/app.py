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
from PySide6.QtGui import QFont

from src.utils.logger import setup_logger
from src.config.settings import AppSettings
from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteImageRepository, SQLiteLicenseRepository, SQLiteIndexedFolderRepository
from src.ai.embedder import OpenCLIPEmbedder
from src.ai.vector_index import FaissIndexManager
from src.core.use_cases.index_images import IndexImagesUseCase
from src.core.use_cases.search_tiles import SearchTilesUseCase
from src.core.use_cases.monitor_folder import FolderMonitorController
from src.core.use_cases.find_duplicates import FindDuplicatesUseCase
from src.core.use_cases.validate_license import ValidateLicenseUseCase
from src.licensing.trial_manager import TrialManager
from src.licensing.validator import LicenseValidator
from src.presentation.viewmodels.indexing_viewmodel import IndexingViewModel
from src.presentation.viewmodels.search_viewmodel import SearchViewModel
from src.presentation.views.main_window import MainWindow
from src.presentation.views.license_view import LicenseView


_app_logger = logging.getLogger("tilevision.app")


def _on_auto_indexed(file_path: str, success: bool, error_message: str) -> None:
    """
    Callback invoked by FolderMonitorController after auto-indexing a
    newly-detected file. Runs on the watchdog background thread, NOT the Qt
    main thread — so this must stay UI-free (logging only) rather than
    touching any QWidget directly, which would not be thread-safe.
    """
    if success:
        _app_logger.info(f"Auto-indexed new file: {file_path}")
    else:
        _app_logger.warning(f"Auto-indexing failed for {file_path}: {error_message}")


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
    app.setOrganizationName("TileVision")
    app.setApplicationVersion("1.0.0")

    # Set modern default font
    default_font = QFont("Segoe UI", 10)
    app.setFont(default_font)

    # ── 4. Construct Data Layer ───────────────────────────────────────────────
    logger.info("Initializing database context...")
    db_context = DatabaseContext(db_path=settings.database_path)

    image_repository = SQLiteImageRepository(db_context=db_context)
    license_repository = SQLiteLicenseRepository(db_context=db_context)
    indexed_folder_repository = SQLiteIndexedFolderRepository(db_context=db_context)

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
    logger.info("Initializing CLIP embedder and FAISS index manager...")
    embedder = OpenCLIPEmbedder(
        model_name=settings.model_name,
        pretrained=settings.pretrained,
    )
    vector_index = FaissIndexManager(
        index_path=settings.index_path,
        dimension=512,
    )

    # ── 8. Construct Use Cases ────────────────────────────────────────────────
    logger.info("Initializing use cases...")
    index_images_use_case = IndexImagesUseCase(
        image_repository=image_repository,
        embedder=embedder,
        vector_index=vector_index,
        thumbnail_dir=settings.thumbnail_dir,
        folder_repository=indexed_folder_repository,
    )
    search_tiles_use_case = SearchTilesUseCase(
        image_repository=image_repository,
        embedder=embedder,
        vector_index=vector_index,
        thumbnail_dir=settings.thumbnail_dir,
    )
    find_duplicates_use_case = FindDuplicatesUseCase(image_repository=image_repository)

    # ── 8b. Warm up the CLIP model and FAISS index now, synchronously, so the
    #        *first* search a user runs is fast rather than paying model-load
    #        cost on that click. Loading is a one-time, ~1-3s startup cost;
    #        after this, both index_images and search_tiles reuse the same
    #        in-memory model/index instances for the lifetime of the process.
    try:
        logger.info("Warming up CLIP model and FAISS index...")
        embedder.load_model()
        vector_index.load_index()
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
    watch_folders = settings.watch_folders
    if watch_folders:
        try:
            logger.info(f"Starting auto folder monitoring for {len(watch_folders)} folder(s)...")
            folder_monitor = FolderMonitorController(
                indexing_use_case=index_images_use_case,
                on_file_indexed_callback=_on_auto_indexed,
            )
            folder_monitor.start_monitoring(watch_folders)
        except Exception as e:
            # Non-fatal: manual indexing/search still work without auto-monitoring.
            logger.error(f"Failed to start folder monitoring (continuing without it): {e}")
            folder_monitor = None
    else:
        logger.info("No watched folders configured — auto folder monitoring not started.")

    # ── 9. Construct ViewModels ───────────────────────────────────────────────
    logger.info("Constructing view models...")
    indexing_viewmodel = IndexingViewModel(use_case=index_images_use_case)
    search_viewmodel = SearchViewModel(use_case=search_tiles_use_case, default_top_k=settings.top_k)

    # ── 10. Launch Main Window ────────────────────────────────────────────────
    logger.info("Launching main application window...")
    main_window = MainWindow(
        indexing_viewmodel=indexing_viewmodel,
        search_viewmodel=search_viewmodel,
        license_details=license_details,
        find_duplicates_use_case=find_duplicates_use_case,
        settings=settings,
        catalog_count_provider=lambda: len(image_repository.get_all()),
    )
    main_window.show()

    logger.info("TileVision AI is running.")

    # ── 11. Run Qt Event Loop ─────────────────────────────────────────────────
    exit_code = app.exec()

    if folder_monitor is not None:
        logger.info("Stopping folder monitor before shutdown...")
        folder_monitor.stop_monitoring()

    logger.info(f"TileVision AI exiting with code: {exit_code}")
    return exit_code
