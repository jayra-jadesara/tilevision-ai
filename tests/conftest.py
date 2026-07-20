"""
Pytest configuration for TileVision AI tests.

Injects lightweight fake `torch` and `open_clip` modules into sys.modules
when the real (heavy) packages aren't installed, so modules that import
them at module load time (e.g. src.ai.embedder) can still be imported for
unit testing the surrounding orchestration logic. Tests that need real
model inference should install the real packages and are skipped/mocked
here via fakes (see FakeEmbedder in individual test files).
"""

import sys
import types

import pytest


def _install_fake_module(name: str, attrs: dict) -> None:
    if name in sys.modules:
        return
    try:
        __import__(name)
        return  # real package is installed, nothing to fake
    except ImportError:
        pass

    fake = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(fake, key, value)
    sys.modules[name] = fake


class _FakeCuda:
    @staticmethod
    def is_available():
        return False


_install_fake_module("torch", {"cuda": _FakeCuda(), "no_grad": lambda: _NoGradCtx()})


class _NoGradCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


_install_fake_module(
    "open_clip",
    {"create_model_and_transforms": lambda *a, **k: (None, None, None)},
)


def pytest_configure():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_collection_modifyitems(config, items):
    """Skip FAISS search tests on macOS GitHub Actions (faiss-cpu abort trap)."""
    import os
    import sys

    if sys.platform != "darwin" or os.environ.get("CI") != "true":
        return

    skip = pytest.mark.skip(
        reason="FAISS search aborts on macOS GitHub Actions runners"
    )
    for item in items:
        if "faiss_search" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def catalogue_master_service(tmp_path):
    """In-memory SQLite export profile service for UI tests."""
    from src.data.db_context import DatabaseContext
    from src.data.sqlite_repository import SQLiteCatalogueProfileRepository
    from src.services.catalogue_master_service import CatalogueMasterService

    db = DatabaseContext(str(tmp_path / "catalogue_test.db"))
    repo = SQLiteCatalogueProfileRepository(db)
    return CatalogueMasterService(
        repository=repo,
        license_customer_name="Test Customer",
    )
