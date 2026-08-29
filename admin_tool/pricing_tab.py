"""Pricing page for the vendor admin tool — edit rates and publish to GitHub."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
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

from github_connect import (
    connect_github_automatically,
    connection_status,
    normalize_pasted_token,
    open_github_token_page,
    publish_target_label,
    save_github_connection,
    try_github_cli_token,
)
from github_pricing_publish import GitHubPublishError, publish_prices_to_github
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
from table_combo import attach_table_combo, combo_value
from vendor_settings import (
    get_pricing_dropdown_options,
    remember_pricing_dropdown_value,
)

_PLAN_IDS = ("1y", "2y", "3y", "4y", "lifetime")


class _GitHubTokenDialog(QDialog):
  """Paste token after browser opens (one-time setup)."""

  def __init__(self, parent: QWidget | None = None) -> None:
      super().__init__(parent)
      self.setWindowTitle("Connect GitHub")
      self.setMinimumWidth(460)
      layout = QVBoxLayout(self)

      intro = QLabel(
          "GitHub opened in your browser.\n"
          "Click Generate token, copy it, then paste below.\n"
          "Saved only on this PC — used when you click Save."
      )
      intro.setWordWrap(True)
      layout.addWidget(intro)

      self._token = QLineEdit()
      self._token.setEchoMode(QLineEdit.EchoMode.Password)
      self._token.setPlaceholderText("ghp_… or github_pat_…")
      layout.addWidget(self._token)

      paste_btn = QPushButton("Paste from clipboard")
      paste_btn.clicked.connect(self._paste_clipboard)
      layout.addWidget(paste_btn)

      buttons = QDialogButtonBox(
          QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
      )
      buttons.accepted.connect(self.accept)
      buttons.rejected.connect(self.reject)
      layout.addWidget(buttons)

        clipboard = QGuiApplication.clipboard().text().strip()
        if clipboard:
            from github_connect import normalize_pasted_token

            token = normalize_pasted_token(clipboard)
            if token:
                self._token.setText(token)

  def _paste_clipboard(self) -> None:
      text = QGuiApplication.clipboard().text().strip()
      if not text:
          return
      from github_connect import normalize_pasted_token

      token = normalize_pasted_token(text)
      if token:
          self._token.setText(token)
      else:
          QMessageBox.warning(
              self,
              "Invalid paste",
              "Clipboard does not contain a GitHub token.\n\n"
              "Copy only the token from GitHub (ghp_… or github_pat_…).",
          )

  def token(self) -> str:
      return self._token.text().strip()


class PricingTab(QWidget):
    """Vendor form to edit live pricing JSON and publish to GitHub."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._base_payload: dict[str, Any] = {}
        self._build_ui()
        self._load_initial_pricing()
        self._refresh_github_status()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        title = QLabel("Pricing")
        title.setObjectName("Title")
        outer.addWidget(title)

        intro = QLabel(
            "Edit rates, click Save to push live to all customers, or Preview PDF first."
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
        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("PrimaryButton")
        self._save_btn.setMinimumWidth(120)
        self._save_btn.clicked.connect(self._on_save)
        actions.addWidget(self._save_btn)

        self._preview_btn = QPushButton("Preview PDF")
        self._preview_btn.setMinimumWidth(120)
        self._preview_btn.clicked.connect(self._on_preview_pdf)
        actions.addWidget(self._preview_btn)
        actions.addStretch()
        outer.addLayout(actions)

        self._status = QLabel("")
        self._status.setObjectName("Hint")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

    def _build_github_group(self) -> QGroupBox:
        box = QGroupBox("GitHub")
        layout = QVBoxLayout(box)
        layout.setSpacing(8)

        self._github_target = QLabel()
        self._github_target.setObjectName("Hint")
        self._github_target.setWordWrap(True)
        layout.addWidget(self._github_target)

        self._github_status = QLabel()
        self._github_status.setObjectName("KeyStatus")
        layout.addWidget(self._github_status)

        row = QHBoxLayout()
        connect_btn = QPushButton("Connect GitHub")
        connect_btn.setObjectName("PrimaryButton")
        connect_btn.clicked.connect(self._on_connect_github)
        row.addWidget(connect_btn)

        test_btn = QPushButton("Test connection")
        test_btn.clicked.connect(self._on_test_connection)
        row.addWidget(test_btn)
        row.addStretch()
        layout.addLayout(row)

        hint = QLabel(
            "First time: Connect GitHub opens your browser with the correct token page. "
            "If GitHub CLI (gh) is installed and logged in, connection is fully automatic."
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return box

    def _refresh_github_status(self) -> None:
        self._github_target.setText(publish_target_label())
        ok, message = connection_status()
        self._github_status.setText(message)
        self._github_status.setProperty("loaded", "true" if ok else "false")
        self._github_status.style().unpolish(self._github_status)
        self._github_status.style().polish(self._github_status)

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
        self._plans_table.verticalHeader().setDefaultSectionSize(40)
        self._plans_table.setMinimumHeight(240)
        layout.addWidget(self._plans_table)
        hint = QLabel(
            "Click a cell to open the dropdown. Pick a value or type a new one and press Enter."
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return box

    def _set_status(self, message: str) -> None:
        self._status.setText(message)

    def _on_connect_github(self) -> None:
        cli_token = try_github_cli_token()
        if cli_token:
            try:
                login = save_github_connection(cli_token)
            except GitHubPublishError as exc:
                QMessageBox.warning(self, "GitHub connection failed", str(exc))
                return
            self._refresh_github_status()
            QMessageBox.information(
                self,
                "GitHub connected",
                f"Connected automatically via GitHub CLI as {login}.",
            )
            return

        open_github_token_page()
        dialog = _GitHubTokenDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self._set_status("GitHub connection cancelled.")
            return
        token = dialog.token()
        if not token:
            QMessageBox.warning(self, "GitHub", "Paste the token from GitHub first.")
            return
        try:
            login = save_github_connection(token)
        except GitHubPublishError as exc:
            QMessageBox.warning(self, "GitHub connection failed", str(exc))
            return
        self._refresh_github_status()
        QMessageBox.information(self, "GitHub connected", f"Connected as {login}.")

    def _on_test_connection(self) -> None:
        ok, message = connection_status()
        if ok:
            QMessageBox.information(self, "GitHub OK", message)
        else:
            QMessageBox.warning(
                self,
                "Not connected",
                f"{message}\n\nClick Connect GitHub first.",
            )
        self._refresh_github_status()

    def _load_initial_pricing(self) -> None:
        try:
            data = fetch_live_prices()
            self._base_payload = data
            self._populate_form(data)
            save_draft(data)
            self._set_status("Loaded live pricing from GitHub.")
            return
        except PricingQuoteError:
            pass
        try:
            self._base_payload = load_template()
            self._populate_form(self._base_payload)
            self._set_status("Loaded local pricing draft.")
        except PricingQuoteError as exc:
            QMessageBox.warning(self, "Pricing load failed", str(exc))

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

        options = get_pricing_dropdown_options()
        plans = data.get("plans") or []
        by_id = {str(p.get("id")): p for p in plans if isinstance(p, dict)}
        self._plans_table.setRowCount(len(_PLAN_IDS))
        for row, plan_id in enumerate(_PLAN_IDS):
            plan = by_id.get(plan_id, {"id": plan_id, "label": plan_id, "price": 0})
            normalized = plan_row_from_dict(plan)
            per_year = normalized.get("effective_label") or normalized.get("effective_per_year")
            badge = normalized.get("badge") or ""

            price_item = QTableWidgetItem(str(normalized["price"]))
            price_item.setTextAlignment(int(Qt.AlignmentFlag.AlignVCenter))
            self._plans_table.setItem(row, 1, price_item)

            attach_table_combo(
                self._plans_table,
                row,
                0,
                options["plan_labels"],
                str(normalized["label"]),
                on_remember=lambda v, c="plan_labels": remember_pricing_dropdown_value(c, v),
            )
            attach_table_combo(
                self._plans_table,
                row,
                2,
                [str(v) for v in options["per_year"]],
                str(per_year or ""),
                on_remember=lambda v, c="per_year": remember_pricing_dropdown_value(c, v),
            )
            attach_table_combo(
                self._plans_table,
                row,
                3,
                options["discount_notes"],
                str(normalized["discount_note"]),
                on_remember=lambda v, c="discount_notes": remember_pricing_dropdown_value(c, v),
            )
            attach_table_combo(
                self._plans_table,
                row,
                4,
                options["badges"],
                str(badge),
                allow_none=True,
                on_remember=lambda v, c="badges": remember_pricing_dropdown_value(c, v),
            )

    def _cell_combo_text(self, row: int, column: int, *, allow_none: bool = False) -> str:
        widget = self._plans_table.cellWidget(row, column)
        from table_combo import TableComboBox

        if isinstance(widget, TableComboBox):
            return combo_value(widget, allow_none=allow_none)
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
            badge_raw = self._cell_combo_text(row, 4, allow_none=True)
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

    def _on_save(self) -> None:
        ok, _ = connection_status()
        if not ok:
            answer = QMessageBox.question(
                self,
                "Connect GitHub",
                "GitHub is not connected yet.\n\n"
                "Connect now? (Required to push live rates to customers)",
            )
            if answer == QMessageBox.StandardButton.Yes:
                self._on_connect_github()
            ok, _ = connection_status()
            if not ok:
                return

        try:
            data = self._collect_payload()
        except PricingQuoteError as exc:
            QMessageBox.warning(self, "Validation failed", str(exc))
            return

        try:
            backup_current(data)
            path = save_draft(data)
            result = publish_prices_to_github(data)
        except GitHubPublishError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return

        self._base_payload = data
        self._set_status(
            f"Saved and published at {datetime.now():%H:%M:%S} — "
            f"customers see new rates on refresh."
        )
        QMessageBox.information(
            self,
            "Saved",
            f"Pricing saved locally and published to GitHub.\n\nDraft: {path}",
        )

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
        self._set_status(f"Preview PDF: {out}")

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
