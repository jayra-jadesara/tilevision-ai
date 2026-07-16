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
    )
    records = ledger.list_licenses("active")
    assert len(records) == 1
    assert records[0].customer_name == "Acme Tiles"

    assert ledger.cancel_license("id-1")
    assert ledger.list_licenses("active") == []
    assert ledger.active_revoked_ids() == ["id-1"]
    assert ledger.is_machine_blocked("machine-abc")
