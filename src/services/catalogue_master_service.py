"""Persistent company profiles for Export Catalogue."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from src.services.company_settings_service import CompanySettingsService

_MASTERS_FILE = Path.home() / ".tilevision_ai" / "catalogue_masters.json"


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
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CatalogueMaster":
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
        )


class CatalogueMasterService:
    """Load/save multiple catalogue export profiles."""

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._path = storage_path or _MASTERS_FILE
        self._last_selected_id: Optional[str] = None
        self._masters: List[CatalogueMaster] = []
        self._load()

    @property
    def masters(self) -> List[CatalogueMaster]:
        return list(self._masters)

    @property
    def last_selected_id(self) -> Optional[str]:
        return self._last_selected_id

    def get(self, master_id: str) -> Optional[CatalogueMaster]:
        for master in self._masters:
            if master.id == master_id:
                return master
        return None

    def set_last_selected(self, master_id: Optional[str]) -> None:
        self._last_selected_id = master_id
        self._persist()

    def add(self, master: CatalogueMaster) -> CatalogueMaster:
        if not master.display_name.strip():
            raise ValueError("Profile name is required.")
        self._masters.append(master)
        self._last_selected_id = master.id
        self._persist()
        return master

    def update(self, master: CatalogueMaster) -> CatalogueMaster:
        if not master.display_name.strip():
            raise ValueError("Profile name is required.")
        for index, existing in enumerate(self._masters):
            if existing.id == master.id:
                self._masters[index] = master
                self._persist()
                return master
        raise KeyError(f"Profile not found: {master.id}")

    def delete(self, master_id: str) -> None:
        self._masters = [m for m in self._masters if m.id != master_id]
        if self._last_selected_id == master_id:
            self._last_selected_id = self._masters[0].id if self._masters else None
        self._persist()

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

    def _load(self) -> None:
        self._masters = []
        self._last_selected_id = None

        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                self._last_selected_id = payload.get("last_selected_id")
                self._masters = [
                    CatalogueMaster.from_dict(item)
                    for item in payload.get("masters", [])
                    if isinstance(item, dict)
                ]
                return
            except Exception:
                pass

        self._migrate_legacy_single_settings()

    def _migrate_legacy_single_settings(self) -> None:
        if self._path != _MASTERS_FILE:
            return
        legacy = CompanySettingsService.load()
        if not any(str(legacy.get(key) or "").strip() for key in legacy):
            return

        name = (legacy.get("company_name") or "Default Profile").strip() or "Default Profile"
        master = CatalogueMaster(
            display_name=name,
            company_name=name,
            logo_path=str(legacy.get("logo_path") or ""),
            email=str(legacy.get("email") or ""),
            phone=str(legacy.get("phone") or ""),
            website=str(legacy.get("website") or ""),
            address=str(legacy.get("address") or ""),
        )
        self._masters = [master]
        self._last_selected_id = master.id
        self._persist()

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_selected_id": self._last_selected_id,
            "masters": [master.to_dict() for master in self._masters],
        }
        with open(self._path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
