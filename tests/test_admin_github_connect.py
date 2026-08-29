"""Tests for GitHub connect helper and table combo."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "admin_tool"))

from github_connect import normalize_pasted_token, publish_target_label, require_github_token  # noqa: E402
from table_combo import TableComboBox, combo_value, populate_combo  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def test_normalize_pasted_token_extracts_ghp():
    token = normalize_pasted_token("  ghp_abc123xyz4567890123456789012345678  ")
    assert token.startswith("ghp_")


def test_normalize_rejects_shell_commands():
    assert normalize_pasted_token(
        "git pull origin cursor/admin-pricing-github-publish-191a\npython .\\admin_tool\\main.py"
    ) == ""


def test_require_github_token_raises_on_garbage():
    with pytest.raises(Exception) as exc:
        require_github_token("git pull origin\npython main.py")
    assert "GitHub token" in str(exc.value)


def test_publish_target_label_contains_repo():
    assert "tilevision-ai" in publish_target_label()


def test_table_combo_populates(qapp):
    combo = TableComboBox()
    populate_combo(combo, ["1 Year", "2 Year", "3 Year"], "2 Year")
    assert combo.count() == 3
    assert combo.currentText() == "2 Year"


def test_combo_value_none_label(qapp):
    combo = TableComboBox()
    populate_combo(combo, ["Best value"], "", allow_none=True)
    assert combo_value(combo, allow_none=True) == ""
