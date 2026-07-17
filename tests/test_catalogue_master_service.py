"""Tests for catalogue master profiles (SQLite-backed)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.db_context import DatabaseContext
from src.data.sqlite_repository import SQLiteCatalogueProfileRepository
from src.services.catalogue_master_service import CatalogueMaster, CatalogueMasterService


def _service(
    tmp_path: Path,
    customer: str = "ABC",
    *,
    legacy_json: Path | None = None,
) -> CatalogueMasterService:
    db = DatabaseContext(str(tmp_path / "test.db"))
    repo = SQLiteCatalogueProfileRepository(db)
    return CatalogueMasterService(
        repository=repo,
        license_customer_name=customer,
        legacy_json_path=legacy_json or (tmp_path / "no_legacy_profiles.json"),
    )


def test_add_list_delete_master(tmp_path: Path) -> None:
    service = _service(tmp_path)

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


def test_profiles_scoped_to_license_customer(tmp_path: Path) -> None:
    db = DatabaseContext(str(tmp_path / "test.db"))
    repo = SQLiteCatalogueProfileRepository(db)

    abc_service = CatalogueMasterService(repository=repo, license_customer_name="ABC")
    xyz_service = CatalogueMasterService(repository=repo, license_customer_name="XYZ")

    abc_service.add(CatalogueMaster(display_name="ABC Profile", company_name="ABC"))
    xyz_service.add(CatalogueMaster(display_name="XYZ Profile", company_name="XYZ"))

    assert len(abc_service.masters) == 1
    assert abc_service.masters[0].display_name == "ABC Profile"
    assert len(xyz_service.masters) == 1
    assert xyz_service.masters[0].display_name == "XYZ Profile"


def test_suggested_pdf_path_uses_master_folder(tmp_path: Path) -> None:
    service = _service(tmp_path)
    master = CatalogueMaster(
        display_name="Demo",
        default_pdf_folder=str(tmp_path / "catalogues"),
    )
    suggested = service.suggested_pdf_path(master, "demo.pdf")
    assert Path(suggested).name == "demo.pdf"
    assert Path(suggested).parent.resolve() == (tmp_path / "catalogues").resolve()


def test_duplicate_profile_name_rejected(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.add(CatalogueMaster(display_name="ABC", company_name="ABC"))

    try:
        service.add(CatalogueMaster(display_name="abc", company_name="ABC Again"))
        assert False, "Expected duplicate profile name to be rejected"
    except ValueError as exc:
        assert "already exists" in str(exc)


def test_ensure_profile_for_customer_only_when_empty(tmp_path: Path) -> None:
    service = _service(tmp_path, customer="ABC")

    created = service.ensure_profile_for_customer("ABC")
    assert created is not None
    assert created.display_name == "ABC"
    assert len(service.masters) == 1

    again = service.ensure_profile_for_customer("ABC")
    assert again is not None
    assert again.id == created.id
    assert len(service.masters) == 1

    service.add(CatalogueMaster(display_name="Other Co", company_name="Other Co"))
    skipped = service.ensure_profile_for_customer("New Customer")
    assert skipped is None
    assert len(service.masters) == 2


def test_export_options_round_trip(tmp_path: Path) -> None:
    service = _service(tmp_path)
    master = CatalogueMaster(
        display_name="Demo",
        include_search_image=False,
        include_image_path=True,
        export_only_selected=True,
        watermark_text="CONFIDENTIAL",
        max_results=24,
    )
    service.add(master)

    reloaded = _service(tmp_path)
    saved = reloaded.get(master.id)
    assert saved is not None
    assert saved.include_search_image is False
    assert saved.include_image_path is True
    assert saved.export_only_selected is True
    assert saved.watermark_text == "CONFIDENTIAL"
    assert saved.max_results == 24


def test_migrate_legacy_json_into_database(tmp_path: Path) -> None:
    legacy_json = tmp_path / "catalogue_masters.json"
    legacy_json.write_text(
        json.dumps(
            {
                "last_selected_id": "profile-1",
                "masters": [
                    {
                        "id": "profile-1",
                        "display_name": "Legacy ABC",
                        "company_name": "Legacy ABC",
                        "email": "legacy@example.com",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    db = DatabaseContext(str(tmp_path / "test.db"))
    repo = SQLiteCatalogueProfileRepository(db)
    service = CatalogueMasterService(
        repository=repo,
        license_customer_name="ABC",
        legacy_json_path=legacy_json,
    )
    service.migrate_legacy_storage_if_needed()

    assert len(service.masters) == 1
    assert service.masters[0].display_name == "Legacy ABC"
    assert service.masters[0].email == "legacy@example.com"
    assert not legacy_json.exists()
    assert legacy_json.with_suffix(".json.migrated").exists()


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

    import src.services.company_settings_service as css

    monkeypatch.setattr(css.CompanySettingsService, "SETTINGS_FILE", legacy)

    service = _service(tmp_path, customer="Legacy Co")
    service.migrate_legacy_storage_if_needed()

    assert len(service.masters) == 1
    assert service.masters[0].company_name == "Legacy Co"
    assert service.masters[0].email == "legacy@example.com"
