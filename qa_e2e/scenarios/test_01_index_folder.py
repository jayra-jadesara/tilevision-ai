"""Customer indexes a showroom catalogue folder."""

from __future__ import annotations

import pytest

from qa_e2e.framework.readiness import probe_readiness

pytestmark = pytest.mark.qa_e2e


def test_index_folder_populates_faiss_and_sqlite(session, driver, catalog_indexed):
    action = session.artifacts.begin("Verify catalog indexed")
    report = probe_readiness(session, require_catalog=True)
    session.human.think(0.2, 0.5)
    driver.goto_index()
    session.artifacts.end(
        action,
        ok=report.catalog_indexed,
        detail=f"faiss={report.faiss_count} sqlite={report.sqlite_count}",
        screenshot_widget=session.main_window,
        metrics={
            "faiss_count": report.faiss_count,
            "sqlite_count": report.sqlite_count,
            "backend": report.faiss_backend,
        },
    )
    assert report.catalog_indexed
    assert report.faiss_count >= 4
    assert report.sqlite_count >= 4
