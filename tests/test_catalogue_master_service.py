"""Tests for catalogue master profiles."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.services.catalogue_master_service import CatalogueMaster, CatalogueMasterService


def test_add_list_delete_master(tmp_path: Path) -> None:
    store = tmp_path / "masters.json"
    service = CatalogueMasterService(storage_path=store)

    master = CatalogueMaster(
        display_name="ABC Ceramic",
        company_name="ABC Ceramic Pvt Ltd",
        email="abc@gmail.com",
        default_pdf_folder=str(tmp_path / "exports"),
    )
    service.add(master)

    assert len(service.masters) == 1
    assert service.get(master.id) is not None
    assert service.last_selected_id == master.id

    service.delete(master.id)
    assert service.masters == []


def test_suggested_pdf_path_uses_master_folder(tmp_path: Path) -> None:
    service = CatalogueMasterService(storage_path=tmp_path / "masters.json")
    master = CatalogueMaster(
        display_name="Demo",
        default_pdf_folder=str(tmp_path / "catalogues"),
    )
    suggested = service.suggested_pdf_path(master, "demo.pdf")
    assert Path(suggested).name == "demo.pdf"
    assert Path(suggested).parent.resolve() == (tmp_path / "catalogues").resolve()


def test_migrate_legacy_company_settings(tmp_path: Path, monkeypatch) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            {
                "company_name": "Legacy Co",
                "email": "legacy@example.com",
            }
        ),
        encoding="utf-8",
    )

    import src.services.catalogue_master_service as cms
    import src.services.company_settings_service as css

    monkeypatch.setattr(css.CompanySettingsService, "SETTINGS_FILE", legacy)
    monkeypatch.setattr(cms, "_MASTERS_FILE", tmp_path / "masters.json")

    service = CatalogueMasterService(storage_path=tmp_path / "masters.json")
    assert len(service.masters) == 1
    assert service.masters[0].company_name == "Legacy Co"
    assert service.masters[0].email == "legacy@example.com"
