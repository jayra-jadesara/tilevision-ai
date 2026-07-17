"""Persistent company profiles for Export Catalogue (SQLite, per license customer)."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from src.data.repository_interface import ICatalogueProfileRepository
from src.services.company_settings_service import CompanySettingsService

logger = logging.getLogger("tilevision.services.catalogue_master_service")

_LEGACY_JSON_FILE = Path.home() / ".tilevision_ai" / "catalogue_masters.json"


@dataclass
class CatalogueMaster:
    """One reusable export catalogue company profile."""

    display_name: str
    company_name: str = ""
    logo_path: str = ""
    email: str = ""
    phone: str = ""
    website: str = ""
    address: str = ""
    default_pdf_folder: str = ""
    include_search_image: bool = True
    include_image_path: bool = False
    export_only_selected: bool = False
    watermark_text: str = ""
    max_results: int = 12
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CatalogueMaster":
        max_results = data.get("max_results", 12)
        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = 12
        max_results = max(1, min(100, max_results))

        return cls(
            id=str(data.get("id") or uuid.uuid4()),
            display_name=str(data.get("display_name") or data.get("company_name") or "Untitled"),
            company_name=str(data.get("company_name") or ""),
            logo_path=str(data.get("logo_path") or ""),
            email=str(data.get("email") or ""),
            phone=str(data.get("phone") or ""),
            website=str(data.get("website") or ""),
            address=str(data.get("address") or ""),
            default_pdf_folder=str(data.get("default_pdf_folder") or ""),
            include_search_image=bool(data.get("include_search_image", True)),
            include_image_path=bool(data.get("include_image_path", False)),
            export_only_selected=bool(data.get("export_only_selected", False)),
            watermark_text=str(data.get("watermark_text") or ""),
            max_results=max_results,
        )


class CatalogueMasterService:
    """Load/save export profiles in SQLite, scoped to the licensed customer."""

    def __init__(
        self,
        repository: ICatalogueProfileRepository,
        license_customer_name: str,
        *,
        legacy_json_path: Optional[Path] = None,
    ) -> None:
        self._repository = repository
        self._license_customer_name = license_customer_name.strip()
        self._legacy_json_path = legacy_json_path or _LEGACY_JSON_FILE
        self._last_selected_id: Optional[str] = None
        self._masters: List[CatalogueMaster] = []
        self._reload()

    @property
    def license_customer_name(self) -> str:
        return self._license_customer_name

    @property
    def masters(self) -> List[CatalogueMaster]:
        return list(self._masters)

    @property
    def last_selected_id(self) -> Optional[str]:
        return self._last_selected_id

    def _reload(self) -> None:
        if not self._license_customer_name:
            self._masters = []
            self._last_selected_id = None
            return
        self._masters = self._repository.list_for_customer(self._license_customer_name)
        self._last_selected_id = self._repository.get_last_selected_id(self._license_customer_name)
        if self._last_selected_id and not self.get(self._last_selected_id):
            self._last_selected_id = self._masters[0].id if self._masters else None
            self._repository.set_last_selected_id(
                self._license_customer_name, self._last_selected_id
            )

    def get(self, master_id: str) -> Optional[CatalogueMaster]:
        for master in self._masters:
            if master.id == master_id:
                return master
        return self._repository.get_by_id(self._license_customer_name, master_id)

    def set_last_selected(self, master_id: Optional[str]) -> None:
        self._last_selected_id = master_id
        self._repository.set_last_selected_id(self._license_customer_name, master_id)

    def add(self, master: CatalogueMaster) -> CatalogueMaster:
        self._require_customer()
        if not master.display_name.strip():
            raise ValueError("Profile name is required.")
        if self.is_display_name_taken(master.display_name):
            raise ValueError(
                f'A profile named "{master.display_name.strip()}" already exists. '
                "Each customer can have only one profile."
            )
        saved = self._repository.add(self._license_customer_name, master)
        self._last_selected_id = saved.id
        self._repository.set_last_selected_id(self._license_customer_name, saved.id)
        self._reload()
        return saved

    def update(self, master: CatalogueMaster) -> CatalogueMaster:
        self._require_customer()
        if not master.display_name.strip():
            raise ValueError("Profile name is required.")
        if self.is_display_name_taken(master.display_name, exclude_id=master.id):
            raise ValueError(
                f'A profile named "{master.display_name.strip()}" already exists. '
                "Each customer can have only one profile."
            )
        saved = self._repository.update(self._license_customer_name, master)
        self._reload()
        return saved

    def delete(self, master_id: str) -> None:
        self._require_customer()
        self._repository.delete(self._license_customer_name, master_id)
        self._reload()
        if self._masters and (
            self._last_selected_id is None or self.get(self._last_selected_id) is None
        ):
            self.set_last_selected(self._masters[0].id)

    def suggested_pdf_path(self, master: CatalogueMaster, default_filename: str) -> str:
        folder = (master.default_pdf_folder or "").strip()
        if folder:
            return str(Path(folder).expanduser() / default_filename)
        return str(Path.home() / default_filename)

    def remember_pdf_folder(self, master_id: str, pdf_path: str) -> None:
        master = self.get(master_id)
        if master is None or not pdf_path:
            return
        master.default_pdf_folder = str(Path(pdf_path).expanduser().resolve().parent)
        self.update(master)

    @staticmethod
    def _normalize_display_name(name: str) -> str:
        return " ".join(name.strip().split()).casefold()

    def is_display_name_taken(
        self, display_name: str, *, exclude_id: Optional[str] = None
    ) -> bool:
        return (
            self.find_by_display_name(display_name, exclude_id=exclude_id) is not None
        )

    def find_by_display_name(
        self, display_name: str, *, exclude_id: Optional[str] = None
    ) -> Optional[CatalogueMaster]:
        if not self._license_customer_name:
            return None
        return self._repository.find_by_display_name(
            self._license_customer_name,
            display_name,
            exclude_id=exclude_id,
        )

    def ensure_profile_for_customer(self, customer_name: str) -> Optional[CatalogueMaster]:
        """Create the first export profile for this licensed customer if none exist."""
        customer_name = customer_name.strip()
        if not customer_name or customer_name != self._license_customer_name:
            return None

        existing = self.find_by_display_name(customer_name)
        if existing is not None:
            self.set_last_selected(existing.id)
            return existing

        if self._masters:
            return None

        return self.add(
            CatalogueMaster(
                display_name=customer_name,
                company_name=customer_name,
            )
        )

    def migrate_legacy_storage_if_needed(self) -> None:
        """
        One-time import from catalogue_masters.json or legacy company settings
        into SQLite for the current licensed customer.
        """
        if not self._license_customer_name:
            return
        if self._repository.count_for_customer(self._license_customer_name) > 0:
            self._archive_legacy_json()
            return

        imported = self._import_legacy_json()
        if not imported:
            imported = self._import_legacy_company_settings()
        if imported:
            self._reload()
            logger.info(
                "Imported export profile(s) into database for customer %s",
                self._license_customer_name,
            )
        self._archive_legacy_json()

    def _import_legacy_json(self) -> bool:
        path = self._legacy_json_path
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False

        masters = [
            CatalogueMaster.from_dict(item)
            for item in payload.get("masters", [])
            if isinstance(item, dict)
        ]
        if not masters:
            return False

        last_selected_id = payload.get("last_selected_id")
        imported_any = False
        for master in masters:
            if self.find_by_display_name(master.display_name):
                continue
            self._repository.add(self._license_customer_name, master)
            imported_any = True

        if imported_any and last_selected_id:
            if self.get(str(last_selected_id)):
                self._repository.set_last_selected_id(
                    self._license_customer_name, str(last_selected_id)
                )
        return imported_any

    def _import_legacy_company_settings(self) -> bool:
        legacy = CompanySettingsService.load()
        if not any(str(legacy.get(key) or "").strip() for key in legacy):
            return False

        name = (legacy.get("company_name") or self._license_customer_name or "Default Profile")
        name = name.strip() or "Default Profile"
        if self.find_by_display_name(name):
            return False

        master = CatalogueMaster(
            display_name=name,
            company_name=name,
            logo_path=str(legacy.get("logo_path") or ""),
            email=str(legacy.get("email") or ""),
            phone=str(legacy.get("phone") or ""),
            website=str(legacy.get("website") or ""),
            address=str(legacy.get("address") or ""),
        )
        self._repository.add(self._license_customer_name, master)
        self._repository.set_last_selected_id(self._license_customer_name, master.id)
        return True

    def _archive_legacy_json(self) -> None:
        path = self._legacy_json_path
        if not path.exists():
            return
        backup = path.with_suffix(".json.migrated")
        try:
            if backup.exists():
                backup.unlink()
            path.rename(backup)
            logger.info("Archived legacy export profile file: %s", backup.name)
        except OSError as exc:
            logger.warning("Could not archive legacy export profile file: %s", exc)

    def _require_customer(self) -> None:
        if not self._license_customer_name:
            raise ValueError(
                "No licensed customer is active. Activate a license before saving export profiles."
            )
