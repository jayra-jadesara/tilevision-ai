"""Tests for vendor license ledger."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "admin_tool"))

from license_ledger import LicenseLedger


def test_record_and_cancel_license(tmp_path):
    ledger = LicenseLedger(db_path=tmp_path / "ledger.db")
    ledger.record_issue(
        license_id="id-1",
        customer_name="Acme Tiles",
        machine_id="machine-abc",
        license_type="1-Year",
        expires_at="2027-01-01",
        license_key="fake-key-data",
        contact="sales@acme.com",
    )
    records = ledger.list_licenses("active", category_filter="official")
    assert len(records) == 1
    assert records[0].license_key == "fake-key-data"

    assert ledger.cancel_license("id-1")
    assert ledger.list_licenses("active") == []
    assert ledger.active_revoked_ids() == ["id-1"]
    assert ledger.is_machine_blocked("machine-abc")


def test_search_and_stats(tmp_path):
    ledger = LicenseLedger(db_path=tmp_path / "ledger.db")
    ledger.record_issue(
        license_id="trial-1",
        customer_name="Beta Shop",
        machine_id="m1",
        license_type="1-Month Trial",
        expires_at="2026-08-01",
        license_key="trial-key",
    )
    ledger.record_issue(
        license_id="paid-1",
        customer_name="Gamma Tiles",
        machine_id="m2",
        license_type="1-Year",
        expires_at="2027-01-01",
        license_key="paid-key",
    )

    trials = ledger.list_licenses(category_filter="trials")
    assert len(trials) == 1
    assert ledger.list_licenses(search_text="Gamma")[0].customer_name == "Gamma Tiles"

    stats = ledger.get_stats()
    assert stats.total == 2
    assert stats.trials_active == 1
    assert stats.official_active == 1


def test_renew_supersedes_old_license(tmp_path):
    ledger = LicenseLedger(db_path=tmp_path / "ledger.db")
    ledger.record_issue(
        license_id="old-1",
        customer_name="Acme",
        machine_id="m1",
        license_type="1-Year",
        expires_at="2026-01-01",
        license_key="old-key",
    )
    ledger.record_issue(
        license_id="new-1",
        customer_name="Acme",
        machine_id="m1",
        license_type="1-Year",
        expires_at="2027-01-01",
        license_key="new-key",
        supersede_license_id="old-1",
    )
    old = ledger.get_license("old-1")
    assert old is not None
    assert old.status == "superseded"
