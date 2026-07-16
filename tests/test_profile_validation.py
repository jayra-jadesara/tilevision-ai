"""Tests for optional export profile field validation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.profile_validation import (
    collect_profile_validation_errors,
    validate_email,
    validate_logo_path,
    validate_phone,
    validate_profile_name,
    validate_website,
)


def test_empty_fields_skip_validation() -> None:
    assert validate_email("") is None
    assert validate_phone("") is None
    assert validate_website("") is None
    assert validate_logo_path("") is None
    assert validate_profile_name("") is not None
    assert collect_profile_validation_errors(
        display_name="",
        email="",
        phone="",
        website="",
        logo_path="",
    ) == ["Profile Name: Profile name is required."]


def test_invalid_email_rejected() -> None:
    assert validate_email("not-an-email") is not None
    assert validate_email("user@company.com") is None


def test_invalid_phone_rejected() -> None:
    assert validate_phone("abc") is not None
    assert validate_phone("12345") is not None
    assert validate_phone("9876543210") is None


def test_invalid_email_examples() -> None:
    assert validate_email("user@") is not None
    assert validate_email("user@company") is not None
    assert validate_email("user@company.com") is None


def test_invalid_website_rejected() -> None:
    assert validate_website("not a url!!!") is not None
    assert validate_website("abc") is not None
    assert validate_website("abc.com") is None
    assert validate_website("https://www.abc.com") is None


def test_logo_extension_and_existence(tmp_path: Path) -> None:
    logo = tmp_path / "logo.png"
    logo.write_bytes(b"fake")
    assert validate_logo_path(str(logo)) is None

    bad = tmp_path / "logo.bmp"
    bad.write_bytes(b"fake")
    assert validate_logo_path(str(bad)) is not None

    assert validate_logo_path(str(tmp_path / "missing.png")) is not None
