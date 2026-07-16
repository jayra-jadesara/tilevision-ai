"""Optional validation for export catalogue company profiles."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

_LOGO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
_WEBSITE_DOMAIN_RE = re.compile(
    r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$"
)


def validate_profile_name(value: str) -> Optional[str]:
    if not value.strip():
        return "Profile name is required."
    return None


def validate_email(value: str) -> Optional[str]:
    if not value.strip():
        return None
    if _EMAIL_RE.match(value.strip()):
        return None
    return "Enter a valid email address (e.g. name@company.com)."


def validate_phone(value: str) -> Optional[str]:
    if not value.strip():
        return None
    digits = re.sub(r"\D", "", value)
    if len(digits) == 10 and digits.isdigit():
        return None
    return "Phone number must be exactly 10 digits."


def validate_website(value: str) -> Optional[str]:
    if not value.strip():
        return None
    candidate = value.strip()
    domain = re.sub(r"^https?://", "", candidate, flags=re.IGNORECASE)
    domain = domain.split("/")[0].split("?")[0]
    if _WEBSITE_DOMAIN_RE.match(domain):
        return None
    return "Enter a valid website (e.g. company.com)."


def validate_logo_path(value: str) -> Optional[str]:
    if not value.strip():
        return None
    path = Path(value.strip()).expanduser()
    if not path.is_file():
        return "Logo file not found."
    if path.suffix.lower() not in _LOGO_EXTENSIONS:
        return "Logo must be JPG, PNG, JPEG, or WEBP."
    return None


def collect_profile_validation_errors(
    *,
    display_name: str = "",
    email: str,
    phone: str,
    website: str,
    logo_path: str,
) -> List[str]:
    errors: List[str] = []
    checks = (
        ("Profile Name", validate_profile_name(display_name)),
        ("Email", validate_email(email)),
        ("Phone", validate_phone(phone)),
        ("Website", validate_website(website)),
        ("Company Logo", validate_logo_path(logo_path)),
    )
    for label, message in checks:
        if message:
            errors.append(f"{label}: {message}")
    return errors
