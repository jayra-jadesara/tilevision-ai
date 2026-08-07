"""
Launch the real TileVision AI stack for human-like E2E testing.

Mirrors ``src.app.build_application`` wiring but:
  - uses an isolated HOME / config directory
  - skips interactive license/setup dialogs
  - returns control to the QA driver (does not block on app.exec forever)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication

from qa_e2e.fixtures.catalog_builder import build_customer_catalog
from qa_e2e.framework.collectors import ArtifactCollector
from qa_e2e.framework.human import HumanSimulator
from qa_e2e.framework.log_capture import LogCapture
from qa_e2e.framework.session import AppSession

ROOT = Path(__file__).resolve().parents[2]


def _ensure_repo_on_path() -> None:
    """
    Make `qa_e2e` importable when running from a checkout.

    When validating a packaged .app, never prepend the checkout root — that
    would shadow frozen `src` with source-tree `src`. Append only.
    """
    import os
    import sys

    if str(ROOT) in sys.path:
        return
    if getattr(sys, "frozen", False) or os.environ.get("TILEVISION_QA_PACKAGED_APP") == "1":
        sys.path.append(str(ROOT))
    else:
        sys.path.insert(0, str(ROOT))


_ensure_repo_on_path()


def _install_dev_license(db_path: str) -> None:
    from src.core.models import LicenseInfo
    from src.data.db_context import DatabaseContext
    from src.data.sqlite_repository import SQLiteLicenseRepository

    payload = {
        "customer_name": "TileVision QA E2E License",
        "expires_at": "2030-12-31",
        "hardware_hash": "*",
        "signature": "DEVELOPMENT_BYPASS_NO_REAL_SIGNATURE",
    }
    key = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
    db = DatabaseContext(db_path=db_path)
    repo = SQLiteLicenseRepository(db_context=db)
    ok = repo.save_license(
        LicenseInfo(
            license_key=key,
            hardware_hash="*",
            customer_name="TileVision QA E2E License",
            expires_at="2030-12-31",
            activated_date=datetime.now(),
        )
    )
    db.close_all()
    if not ok:
        raise RuntimeError("Failed to install QA development license")


def prepare_isolated_home(work_dir: Path) -> Path:
    """
    Point Path.home() / ~/.tilevision_ai at an isolated workspace.

    Production code reads ``Path.home() / ".tilevision_ai"`` — we set HOME
    (and USERPROFILE on Windows) so no production edits are required.
    """
    home = work_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(home)
    os.environ["USERPROFILE"] = str(home)
    os.environ["TILEVISION_DEV_MODE"] = "1"
    os.environ.setdefault("TILEVISION_LOG_LEVEL", "INFO")
    os.environ.setdefault("TILEVISION_PROFILE", "1")
    os.environ.setdefault("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    # Avoid update-check popups during customer simulation.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return home


def launch_customer_app(
    *,
    work_dir: Path,
    artifact_dir: Path,
    human_seed: Optional[int] = 42,
    human_speed: float = 1.5,
    catalog_tiles: int = 12,
) -> AppSession:
    """Build and show the real MainWindow with warm DINOv2 + FAISS + SQLite."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    home = prepare_isolated_home(work_dir)
    catalog_root = work_dir / "customer_data"
    manifest = build_customer_catalog(catalog_root, tile_count=catalog_tiles)

    from src.utils.logger import setup_logger
    from src.config.settings import AppSettings
    from src.version import APP_VERSION

    root_logger = setup_logger(
        name="tilevision",
        log_file_name="tilevision.log",
        log_level=logging.INFO,
    )
    logs = LogCapture(level=logging.DEBUG)
    logs.attach("tilevision")

    settings = AppSettings(config_dir=home / ".tilevision_ai")
    settings.setup_wizard_completed = True
    settings.check_for_updates = False
    settings.index_backend = "flat_ip"
    settings.top_k = 10
    # Keep AI work single-threaded in QA — avoids OpenMP/QThread stalls on
    # Mac Intel CI hosts while still using the real DINOv2 stack.
    settings.preprocess_workers = 1
    settings.index_batch_size = 2
    settings.save()

    _install_dev_license(settings.database_path)

    from src.ai.preprocess.sam2_backend import configure_sam2_from_settings

    configure_sam2_from_settings(settings.enable_sam2_precise_crop)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("TileVision AI")
    app.setOrganizationName("JD Software")
    app.setApplicationVersion(APP_VERSION)

    # Headless CI: never block forever on customer QMessageBox dialogs.
    from qa_e2e.framework.dialogs import install_dialog_auto_dismiss
    from qa_e2e.framework.qthread_patch import install_python_thread_workers

    install_dialog_auto_dismiss(app)
    install_python_thread_workers()

    from src.utils.platform_info import app_icon_path, default_ui_font_family
    from src.theme.theme_manager import get_app_stylesheet

    icon_path = app_icon_path()
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))
    app.setFont(QFont(default_ui_font_family(), 10))
    app.setStyleSheet(get_app_stylesheet(settings.theme))

    from src.data.db_context import DatabaseContext
    from src.data.sqlite_repository import (
        SQLiteImageRepository,
        SQLiteLicenseRepository,
        SQLiteIndexedFolderRepository,
        SQLiteSearchHistoryRepository,
        SQLiteActivityLogRepository,
        SQLiteCatalogueProfileRepository,
    )
    from src.licensing.validator import LicenseValidator
    from src.core.use_cases.validate_license import ValidateLicenseUseCase
    from src.ai.vector_index import FaissIndexManager
    from src.ai.embedder import DINOv2Embedder
    from src.ai.feature_extractor import FeatureExtractor
    from src.ai.gpu_info import configure_mps_fallback
    from src.ai.preprocess.image_preprocessor import ImagePreprocessor
    from src.config.indexing_performance import IndexingPerformanceConfig
    from src.ai.index_backends import BackendParams, IndexBackend
    from src.core.use_cases.index_images import IndexImagesUseCase
    from src.core.use_cases.search_tiles import SearchTilesUseCase
    from src.core.use_cases.find_duplicates import FindDuplicatesUseCase
    from src.presentation.viewmodels.indexing_viewmodel import IndexingViewModel
    from src.presentation.viewmodels.search_viewmodel import SearchViewModel
    from src.presentation.views.main_window import MainWindow, DashboardDataProviders
    from src.presentation.auto_index_notifier import AutoIndexNotifier
    from src.services.catalogue_master_service import CatalogueMasterService
    from src.utils.diagnostics import log_startup_diagnostics
    from src.utils.pipeline_timing import profiling_enabled
    from src.ai.feature_versions import CURRENT_EMBEDDING_DIMENSION, CURRENT_EMBEDDING_MODEL
    from src.core.compatibility import run_compatibility_check
    import time as _time

    logger = logging.getLogger("tilevision.app")
    logger.info("QA E2E harness launching customer session (home=%s)", home)

    db_context = DatabaseContext(db_path=settings.database_path)
    image_repository = SQLiteImageRepository(db_context=db_context)
    license_repository = SQLiteLicenseRepository(db_context=db_context)
    indexed_folder_repository = SQLiteIndexedFolderRepository(db_context=db_context)
    search_history_repository = SQLiteSearchHistoryRepository(db_context=db_context)
    activity_log_repository = SQLiteActivityLogRepository(db_context=db_context)
    catalogue_profile_repository = SQLiteCatalogueProfileRepository(db_context=db_context)

    license_validator = LicenseValidator()
    validate_license_use_case = ValidateLicenseUseCase(
        license_repository=license_repository,
        validator=license_validator,
    )
    license_details = validate_license_use_case.verify_existing_license()
    if license_details is None:
        raise RuntimeError("QA license missing after install — cannot launch like a customer")

    customer_name = str(license_details.get("customer_name") or "").strip()
    catalogue_master_service = CatalogueMasterService(
        repository=catalogue_profile_repository,
        license_customer_name=customer_name,
    )
    if customer_name:
        catalogue_master_service.ensure_profile_for_customer(customer_name)

    configure_mps_fallback()
    embedder = DINOv2Embedder(device_preference=settings.inference_device)
    indexing_perf = IndexingPerformanceConfig.from_settings(settings, use_gpu=embedder.using_gpu)
    ImagePreprocessor.configure(max_decode_edge=indexing_perf.max_decode_edge)
    feature_extractor = FeatureExtractor(
        embedder=embedder,
        preprocess_workers=indexing_perf.preprocess_workers,
    )
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
    find_duplicates_use_case = FindDuplicatesUseCase(
        image_repository=image_repository, vector_index=vector_index
    )

    t0 = _time.perf_counter()
    feature_extractor.load_model()
    model_ms = (_time.perf_counter() - t0) * 1000.0
    t1 = _time.perf_counter()
    vector_index.load_index()
    faiss_ms = (_time.perf_counter() - t1) * 1000.0

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

    compatibility_report = run_compatibility_check(
        database_path=settings.database_path,
        index_path=settings.index_path,
        expected_backend=backend,
        feature_status_provider=image_repository.get_feature_version_status,
        catalog_size=vector_index.get_total_count(),
    )
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
            "qa_e2e": True,
        }
    )

    indexing_viewmodel = IndexingViewModel(
        use_case=index_images_use_case, activity_log_repository=activity_log_repository
    )

    def _on_search_busy_changed(busy: bool) -> None:
        if busy:
            indexing_viewmodel.pause_for_search()
        else:
            indexing_viewmodel.resume_after_search()

    search_viewmodel = SearchViewModel(
        use_case=search_tiles_use_case,
        default_top_k=settings.top_k,
        search_history_repository=search_history_repository,
        activity_log_repository=activity_log_repository,
        on_search_busy_changed=_on_search_busy_changed,
    )

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

    def _diagnostics_info() -> dict:
        return {
            "app_version": APP_VERSION,
            "catalog_size": vector_index.get_total_count(),
            "faiss_type": vector_index.index_type_name(),
            "index_backend": vector_index.active_backend().value,
            "database": settings.database_path,
            "device": embedder.runtime_info.summary_for_ui(),
        }

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
        indexed_folders_provider=lambda: [
            f.folder_path for f in indexed_folder_repository.get_all_folders()
        ],
        feature_version_provider=image_repository.get_feature_version_status,
        gpu_info_provider=lambda: embedder.runtime_info,
        diagnostics_info_provider=_diagnostics_info,
        on_watch_folders_changed=lambda: None,
        on_check_updates=None,
    )
    # Keep AutoIndexNotifier referenced so GC doesn't surprise us.
    _ = AutoIndexNotifier()
    main_window.resize(1280, 840)
    main_window.show()
    app.processEvents()

    artifacts = ArtifactCollector(artifact_dir)
    artifacts.note(f"model_warmup_ms={model_ms:.1f}")
    artifacts.note(f"faiss_warmup_ms={faiss_ms:.1f}")
    artifacts.note(f"home={home}")
    artifacts.note(f"catalog={catalog_root / 'catalog'}")

    return AppSession(
        app=app,
        main_window=main_window,
        settings=settings,
        home_dir=home,
        data_dir=home / ".tilevision_ai",
        catalog_dir=catalog_root / "catalog",
        query_dir=catalog_root / "queries",
        artifacts=artifacts,
        logs=logs,
        human=HumanSimulator(seed=human_seed, speed=human_speed),
        search_viewmodel=search_viewmodel,
        indexing_viewmodel=indexing_viewmodel,
        search_use_case=search_tiles_use_case,
        index_use_case=index_images_use_case,
        vector_index=vector_index,
        image_repository=image_repository,
        feature_extractor=feature_extractor,
        db_context=db_context,
        embedder=embedder,
        expected_manifest=manifest,
    )
