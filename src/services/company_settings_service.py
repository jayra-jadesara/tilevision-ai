import json
from pathlib import Path


class CompanySettingsService:
    SETTINGS_FILE = Path.home() / ".tilevision_company.json"

    DEFAULTS = {
        "company_name": "",
        "logo_path": "",
        "email": "",
        "phone": "",
        "website": "",
        "address": "",
    }

    @classmethod
    def load(cls):
        if not cls.SETTINGS_FILE.exists():
            return cls.DEFAULTS.copy()

        try:
            with open(cls.SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            settings = cls.DEFAULTS.copy()
            settings.update(data)
            return settings

        except Exception:
            return cls.DEFAULTS.copy()

    @classmethod
    def save(cls, settings: dict):
        cls.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(cls.SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)