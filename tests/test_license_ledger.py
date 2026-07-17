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


def test_unblock_machine_allows_new_keys_but_keeps_revocation(tmp_path):
    ledger = LicenseLedger(db_path=tmp_path / "ledger.db")
    ledger.record_issue(
        license_id="id-1",
        customer_name="Acme Tiles",
        machine_id="machine-abc",
        license_type="15-Day Trial",
        expires_at="2026-08-01",
        license_key="fake-key-data",
    )
    ledger.cancel_license("id-1")
    assert ledger.is_machine_blocked("machine-abc")
    assert ledger.active_revoked_ids() == ["id-1"]

    assert ledger.unblock_machine("machine-abc", reason="Customer approved")
    assert not ledger.is_machine_blocked("machine-abc")
    assert ledger.is_machine_unblocked("machine-abc")
    assert ledger.active_revoked_ids() == ["id-1"]

    ledger.record_issue(
        license_id="id-2",
        customer_name="Acme Tiles",
        machine_id="machine-abc",
        license_type="15-Day Trial",
        expires_at="2026-09-01",
        license_key="new-key",
    )
    assert ledger.get_license("id-2").status == "active"


def test_cancel_after_unblock_blocks_again(tmp_path):
    ledger = LicenseLedger(db_path=tmp_path / "ledger.db")
    ledger.record_issue(
        license_id="id-1",
        customer_name="Acme",
        machine_id="m1",
        license_type="1-Year",
        expires_at="2027-01-01",
        license_key="key-1",
    )
    ledger.cancel_license("id-1")
    ledger.unblock_machine("m1")
    assert not ledger.is_machine_blocked("m1")

    ledger.record_issue(
        license_id="id-2",
        customer_name="Acme",
        machine_id="m1",
        license_type="1-Year",
        expires_at="2028-01-01",
        license_key="key-2",
    )
    ledger.cancel_license("id-2")
    assert ledger.is_machine_blocked("m1")
    assert not ledger.is_machine_unblocked("m1")


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


def test_list_current_per_machine_shows_latest_active_only(tmp_path):
    ledger = LicenseLedger(db_path=tmp_path / "ledger.db")
    ledger.record_issue(
        license_id="trial-1",
        customer_name="Acme",
        machine_id="m1",
        license_type="1-Month Trial",
        expires_at="2026-08-01",
        license_key="trial-key",
    )
    ledger.record_issue(
        license_id="full-1",
        customer_name="Acme",
        machine_id="m1",
        license_type="1-Year",
        expires_at="2027-01-01",
        license_key="full-key",
        supersede_license_id="trial-1",
    )
    ledger.record_issue(
        license_id="other-pc",
        customer_name="Acme",
        machine_id="m2",
        license_type="1-Year",
        expires_at="2027-01-01",
        license_key="other-key",
    )

    current = ledger.list_current_per_machine()
    assert len(current) == 2
    by_machine = {rec.machine_id: rec for rec in current}
    assert by_machine["m1"].license_id == "full-1"
    assert by_machine["m2"].license_id == "other-pc"
