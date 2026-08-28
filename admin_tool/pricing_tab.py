"""Pricing page for the vendor admin tool — edit rates and publish to GitHub."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from github_pricing_publish import (
    GitHubPublishError,
    publish_prices_to_github,
    verify_github_token,
)
from pricing_manager import (
    apply_editable_fields,
    backup_current,
    fetch_live_prices,
    load_template,
    plan_row_from_dict,
    plans_to_publish_rows,
    save_draft,
)
from src.services.pricing_quote_service import PricingQuoteError, render_pricing_pdf
from vendor_settings import (
    DEFAULT_GITHUB_BRANCH,
    DEFAULT_GITHUB_REPO,
    ensure_github_defaults,
    get_github_branch,
    get_github_repo,
    get_github_token,
    get_pricing_dropdown_options,
    remember_pricing_dropdown_value,
    save_vendor_settings,
)

_PLAN_IDS = ("1y", "2y", "3y", "4y", "lifetime")
_DROPDOWN_KEYS = ("plan_labels", "per_year", "discount_notes", "badges")


def _make_editable_combo(
    options: list[str],
    current: str,
    *,
    on_remember: Callable[[str], None],
) -> QComboBox:
    combo = QComboBox()
    combo.setEditable(True)
    combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    seen: set[str] = set()
    for value in options:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        combo.addItem(text)
    combo.setCurrentText(current)
    combo.lineEdit().setPlaceholderText("Select or type…")  # type: ignore[union-attr]
    combo.currentTextChanged.connect(lambda text: on_remember(text.strip()))
    return combo


class PricingTab(QWidget):
    """Vendor form to edit live pricing JSON and publish to GitHub."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._base_payload: dict[str, Any] = {}
        ensure_github_defaults()
        self._build_ui()
        self._load_template_into_form()
        self._refresh_github_status()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        title = QLabel("Pricing")
        title.setObjectName("Title")
        outer.addWidget(title)

        intro = QLabel(
            "Edit license pricing and publish to GitHub. "
            "Customer apps download the live JSON automatically — no installer rebuild."
        )
        intro.setObjectName("Hint")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        layout.addWidget(self._build_github_group())
        layout.addWidget(self._build_quote_group())
        layout.addWidget(self._build_vendor_group())
        layout.addWidget(self._build_plans_group())
        layout.addStretch()
        scroll.setWidget(page)
        outer.addWidget(scroll, stretch=1)

        actions = QHBoxLayout()
        self._load_live_btn = QPushButton("Load Live")
        self._load_live_btn.clicked.connect(self._on_load_live)
        actions.addWidget(self._load_live_btn)

        self._save_draft_btn = QPushButton("Save Draft")
        self._save_draft_btn.clicked.connect(self._on_save_draft)
        actions.addWidget(self._save_draft_btn)

        self._preview_btn = QPushButton("Preview PDF")
        self._preview_btn.setObjectName("PrimaryButton")
        self._preview_btn.clicked.connect(self._on_preview_pdf)
        actions.addWidget(self._preview_btn)

        self._publish_btn = QPushButton("Publish to GitHub")
        self._publish_btn.setObjectName("PrimaryButton")
        self._publish_btn.clicked.connect(self._on_publish)
        actions.addWidget(self._publish_btn)
        actions.addStretch()
        outer.addLayout(actions)

        self._status = QLabel("")
        self._status.setObjectName("Hint")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

    def _build_github_group(self) -> QGroupBox:
        box = QGroupBox("GitHub publish")
        form = QFormLayout(box)

        self._github_target = QLabel()
        self._github_target.setObjectName("Hint")
        self._github_target.setWordWrap(True)
        form.addRow("Target", self._github_target)

        self._github_token = QLineEdit()
        self._github_token.setEchoMode(QLineEdit.EchoMode.Password)
        self._github_token.setPlaceholderText("Paste token once — saved on this PC only")
        token = get_github_token()
        if token:
            self._github_token.setText(token)
        form.addRow("Access token", self._github_token)

        row = QHBoxLayout()
        test_btn = QPushButton("Test connection")
        test_btn.setObjectName("PrimaryButton")
        test_btn.clicked.connect(self._on_test_github)
        row.addWidget(test_btn)
        row.addStretch()
        form.addRow(row)

        hint = QLabel(
            "Repository and branch are configured automatically. "
            "Create a token at GitHub → Settings → Developer settings → "
            "Personal access tokens (Contents: Read and write)."
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        form.addRow(hint)
        return box

    def _refresh_github_status(self) -> None:
        repo = get_github_repo() or DEFAULT_GITHUB_REPO
        branch = get_github_branch() or DEFAULT_GITHUB_BRANCH
        self._github_target.setText(f"{repo}  ·  branch {branch}  ·  pricing/prices.json")

    def _build_quote_group(self) -> QGroupBox:
        box = QGroupBox("Quote text")
        form = QFormLayout(box)
        self._location = QLineEdit()
        self._hero_title = QLineEdit()
        self._hero_body = QPlainTextEdit()
        self._hero_body.setFixedHeight(72)
        self._pricing_heading = QLineEdit()
        self._taxes_line = QLineEdit()
        form.addRow("Location", self._location)
        form.addRow("Hero title", self._hero_title)
        form.addRow("Hero body", self._hero_body)
        form.addRow("Pricing heading", self._pricing_heading)
        form.addRow("Taxes line", self._taxes_line)
        return box

    def _build_vendor_group(self) -> QGroupBox:
        box = QGroupBox("Vendor contact")
        form = QFormLayout(box)
        self._vendor_name = QLineEdit()
        self._vendor_email = QLineEdit()
        self._vendor_phone = QLineEdit()
        self._vendor_phone_display = QLineEdit()
        form.addRow("Company name", self._vendor_name)
        form.addRow("Email", self._vendor_email)
        form.addRow("Phone", self._vendor_phone)
        form.addRow("Phone display line", self._vendor_phone_display)
        return box

    def _build_plans_group(self) -> QGroupBox:
        box = QGroupBox("License plans (INR)")
        layout = QVBoxLayout(box)
        self._plans_table = QTableWidget(0, 5)
        self._plans_table.setHorizontalHeaderLabels(
            ["Plan", "Price (INR)", "Per year", "Discount note", "Badge"]
        )
        header = self._plans_table.horizontalHeader()
        header.setStretchLastSection(True)
        self._plans_table.verticalHeader().setVisible(False)
        self._plans_table.setMinimumHeight(220)
        layout.addWidget(self._plans_table)
        hint = QLabel(
            "Use dropdowns for Plan, Per year, Discount, and Badge. "
            "Type a new value in any dropdown to add it to the list."
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return box

    def _dropdown_options(self) -> dict[str, list[str]]:
        return get_pricing_dropdown_options()

    def _set_status(self, message: str) -> None:
        self._status.setText(message)

    def _persist_token(self) -> None:
        token = self._github_token.text().strip()
        if token:
            save_vendor_settings(github_token=token)
        ensure_github_defaults()

    def _on_test_github(self) -> None:
        self._persist_token()
        try:
            login = verify_github_token(token=self._github_token.text().strip())
        except GitHubPublishError as exc:
            QMessageBox.warning(self, "GitHub connection failed", str(exc))
            return
        QMessageBox.information(
            self,
            "GitHub OK",
            f"Connected as {login}.\n\n"
            f"Ready to publish to {get_github_repo()} ({get_github_branch()}).",
        )
        self._set_status(f"GitHub connection OK ({login}).")

    def _load_template_into_form(self) -> None:
        try:
            self._base_payload = load_template()
        except PricingQuoteError as exc:
            QMessageBox.warning(self, "Pricing load failed", str(exc))
            return
        self._populate_form(self._base_payload)
        self._set_status("Loaded pricing draft/template.")

    def _populate_form(self, data: dict[str, Any]) -> None:
        self._location.setText(str(data.get("location", "")))
        self._hero_title.setText(str(data.get("hero_title", "")))
        self._hero_body.setPlainText(str(data.get("hero_body", "")))
        self._pricing_heading.setText(str(data.get("pricing_heading", "")))
        footer = data.get("footer") or {}
        self._taxes_line.setText(str(footer.get("taxes", "")))

        vendor = data.get("vendor") or {}
        self._vendor_name.setText(str(vendor.get("name", "")))
        self._vendor_email.setText(str(vendor.get("email", "")))
        self._vendor_phone.setText(str(vendor.get("phone", "")))
        self._vendor_phone_display.setText(str(vendor.get("phone_display", "")))

        options = self._dropdown_options()
        plans = data.get("plans") or []
        by_id = {str(p.get("id")): p for p in plans if isinstance(p, dict)}
        self._plans_table.setRowCount(len(_PLAN_IDS))
        for row, plan_id in enumerate(_PLAN_IDS):
            plan = by_id.get(plan_id, {"id": plan_id, "label": plan_id, "price": 0})
            normalized = plan_row_from_dict(plan)
            per_year = normalized.get("effective_label") or normalized.get("effective_per_year")
            badge = normalized.get("badge") or ""

            self._plans_table.setItem(
                row, 1, QTableWidgetItem(str(normalized["price"]))
            )
            self._plans_table.setCellWidget(
                row,
                0,
                _make_editable_combo(
                    options["plan_labels"],
                    str(normalized["label"]),
                    on_remember=lambda v: remember_pricing_dropdown_value("plan_labels", v),
                ),
            )
            self._plans_table.setCellWidget(
                row,
                2,
                _make_editable_combo(
                    [str(v) for v in options["per_year"]],
                    str(per_year or ""),
                    on_remember=lambda v: remember_pricing_dropdown_value("per_year", v),
                ),
            )
            self._plans_table.setCellWidget(
                row,
                3,
                _make_editable_combo(
                    options["discount_notes"],
                    str(normalized["discount_note"]),
                    on_remember=lambda v: remember_pricing_dropdown_value("discount_notes", v),
                ),
            )
            self._plans_table.setCellWidget(
                row,
                4,
                _make_editable_combo(
                    options["badges"],
                    str(badge),
                    on_remember=lambda v: remember_pricing_dropdown_value("badges", v),
                ),
            )

    def _cell_combo_text(self, row: int, column: int) -> str:
        widget = self._plans_table.cellWidget(row, column)
        if isinstance(widget, QComboBox):
            return widget.currentText().strip()
        item = self._plans_table.item(row, column)
        return item.text().strip() if item else ""

    def _collect_plans_from_table(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row, plan_id in enumerate(_PLAN_IDS):
            label = self._cell_combo_text(row, 0) or plan_id
            price_item = self._plans_table.item(row, 1)
            price_text = price_item.text().strip() if price_item else "0"
            per_year_text = self._cell_combo_text(row, 2)
            discount = self._cell_combo_text(row, 3) or "-"
            badge_raw = self._cell_combo_text(row, 4)
            badge = badge_raw or None

            try:
                price = int(round(float(price_text.replace(",", ""))))
            except ValueError as exc:
                raise PricingQuoteError(f"Invalid price for {label}: {price_text}") from exc

            plan: dict[str, Any] = {
                "id": plan_id,
                "label": label,
                "price": price,
                "discount_note": discount or "-",
                "badge": badge,
            }
            if plan_id == "lifetime" or per_year_text.lower() in {"one-time", "onetime", "once"}:
                plan["effective_label"] = per_year_text or "One-time"
                plan["effective_per_year"] = None
            elif per_year_text:
                try:
                    plan["effective_per_year"] = int(
                        round(float(per_year_text.replace(",", "")))
                    )
                except ValueError as exc:
                    raise PricingQuoteError(
                        f"Invalid per-year value for {label}: {per_year_text}"
                    ) from exc
            rows.append(plan)
        return plans_to_publish_rows(rows)

    def _collect_payload(self) -> dict[str, Any]:
        plans = self._collect_plans_from_table()
        return apply_editable_fields(
            self._base_payload,
            location=self._location.text(),
            hero_title=self._hero_title.text(),
            hero_body=self._hero_body.toPlainText(),
            pricing_heading=self._pricing_heading.text(),
            taxes_line=self._taxes_line.text(),
            vendor_name=self._vendor_name.text(),
            vendor_email=self._vendor_email.text(),
            vendor_phone=self._vendor_phone.text(),
            vendor_phone_display=self._vendor_phone_display.text(),
            plans=plans,
        )

    def _on_load_live(self) -> None:
        try:
            data = fetch_live_prices()
        except PricingQuoteError as exc:
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        self._base_payload = data
        self._populate_form(data)
        save_draft(data)
        self._set_status("Loaded live pricing from GitHub and saved as draft.")

    def _on_save_draft(self) -> None:
        try:
            data = self._collect_payload()
        except PricingQuoteError as exc:
            QMessageBox.warning(self, "Validation failed", str(exc))
            return
        self._base_payload = data
        path = save_draft(data)
        self._set_status(f"Draft saved to {path}")

    def _on_preview_pdf(self) -> None:
        try:
            data = self._collect_payload()
        except PricingQuoteError as exc:
            QMessageBox.warning(self, "Validation failed", str(exc))
            return
        out = Path(tempfile.gettempdir()) / "TileVision_Admin_Pricing_Preview.pdf"
        try:
            render_pricing_pdf(data, output_path=out)
        except PricingQuoteError as exc:
            QMessageBox.warning(self, "PDF failed", str(exc))
            return
        self._open_path(out)
        self._set_status(f"Preview PDF written to {out}")

    def _open_path(self, path: Path) -> None:
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", str(path)], check=False)
            elif sys.platform.startswith("win"):
                subprocess.run(["cmd", "/c", "start", "", str(path)], check=False)
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except Exception:
            pass

    def _on_publish(self) -> None:
        self._persist_token()
        try:
            data = self._collect_payload()
        except PricingQuoteError as exc:
            QMessageBox.warning(self, "Validation failed", str(exc))
            return

        repo = get_github_repo() or DEFAULT_GITHUB_REPO
        branch = get_github_branch() or DEFAULT_GITHUB_BRANCH
        confirm = QMessageBox.question(
            self,
            "Publish pricing",
            "Publish updated pricing to GitHub?\n\n"
            f"Repository: {repo}\n"
            f"Branch: {branch}\n\n"
            "All customers will see new rates when they open Pricing in the app.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            backup_current(data)
            result = publish_prices_to_github(data, token=self._github_token.text().strip())
        except GitHubPublishError as exc:
            QMessageBox.warning(self, "Publish failed", str(exc))
            return

        self._base_payload = data
        save_draft(data)
        paths = "\n".join(f"  • {p}" for p in result.updated_paths)
        QMessageBox.information(
            self,
            "Published",
            f"Pricing published to {result.repo} ({result.branch}).\n\nUpdated:\n{paths}",
        )
        self._set_status(
            f"Published at {datetime.now():%Y-%m-%d %H:%M:%S} — customers get live rates on refresh."
        )
