"""Write release_report.html / .json / .pdf for customer-ship gate inspection."""

from __future__ import annotations

import html
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _esc(text: Any) -> str:
    return html.escape("" if text is None else str(text))


def write_release_reports(
    *,
    out_dir: Path,
    environment: Dict[str, Any],
    gates: List[Dict[str, Any]],
    scenarios: List[Dict[str, Any]],
    logs: List[str],
    final_pass: bool,
    title: str = "TileVision AI — Release Validation Report",
    resources: Optional[List[Dict[str, Any]]] = None,
    actions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "title": title,
        "verdict": "PASS" if final_pass else "FAIL",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "environment": environment,
        "gates": gates,
        "scenarios": scenarios,
        "failed_scenarios": [s for s in scenarios if not s.get("ok")],
        "resources": resources or [],
        "actions": actions or [],
        "log_excerpt": logs[-500:],
        "policy": {
            "overall_pass_requires": "ALL scenarios pass",
            "mocks_allowed": False,
            "stack": ["PySide6", "DINOv2", "SAM2", "SQLite", "FAISS", "HybridRerank"],
        },
    }

    json_path = out_dir / "release_report.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    html_path = _write_html(out_dir / "release_report.html", payload)
    pdf_path = _write_pdf(out_dir / "release_report.pdf", payload)
    return {"json": json_path, "html": html_path, "pdf": pdf_path}


def _write_html(path: Path, payload: Dict[str, Any]) -> Path:
    verdict = payload["verdict"]
    color = "#0a7a32" if verdict == "PASS" else "#b00020"
    gate_rows = "".join(
        f"<tr class='{'ok' if g.get('ok') else 'bad'}'><td>{_esc(g.get('id'))}</td>"
        f"<td>{_esc(g.get('name'))}</td><td>{_esc(g.get('detail'))}</td></tr>"
        for g in payload.get("gates", [])
    )
    scenario_rows = "".join(
        f"<tr class='{'ok' if s.get('ok') else 'bad'}'>"
        f"<td>{_esc(s.get('id'))}</td><td>{_esc(s.get('name'))}</td>"
        f"<td>{'PASS' if s.get('ok') else 'FAIL'}</td>"
        f"<td>{float(s.get('duration_s') or 0):.2f}s</td>"
        f"<td>{_esc(s.get('detail') or s.get('error'))}</td>"
        f"<td>{('<img src=\"'+_esc(s.get('screenshot'))+'\"/>') if s.get('screenshot') else ''}</td>"
        f"</tr>"
        for s in payload.get("scenarios", [])
    )
    fails = payload.get("failed_scenarios") or []
    fail_block = "".join(
        f"<li><strong>{_esc(f.get('id'))}</strong> {_esc(f.get('name'))}: "
        f"<pre>{_esc(f.get('stacktrace') or f.get('error') or f.get('detail'))}</pre></li>"
        for f in fails
    ) or "<li>None</li>"

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{_esc(payload['title'])} — {verdict}</title>
<style>
body{{font-family:Georgia,serif;margin:0;background:#f6f3ec;color:#1c1917}}
header{{padding:40px 48px;border-bottom:1px solid #e7e0d5;background:#fffdf8}}
.verdict{{display:inline-block;padding:6px 14px;border-radius:999px;color:#fff;background:{color};font-weight:700}}
main{{padding:28px 48px 80px;max-width:1200px}}
table{{width:100%;border-collapse:collapse;background:#fffdf8}}
th,td{{border:1px solid #e7e0d5;padding:8px;text-align:left;vertical-align:top;font-size:14px}}
tr.ok td{{background:#f3faf5}} tr.bad td{{background:#fff1f1}}
img{{max-width:220px;border:1px solid #e7e0d5}}
pre{{white-space:pre-wrap;background:#f8f4ec;padding:10px;border-radius:8px;font-size:12px}}
</style></head><body>
<header>
  <div class="verdict">{verdict}</div>
  <h1>{_esc(payload['title'])}</h1>
  <p>{_esc(payload.get('generated_at'))} · {_esc(payload.get('platform'))} · {_esc(payload.get('machine'))}</p>
  <p><strong>Policy:</strong> overall PASS only if every scenario passes. No mocks.</p>
</header>
<main>
<section><h2>Environment</h2><pre>{_esc(json.dumps(payload.get('environment'), indent=2, default=str))}</pre></section>
<section><h2>Gates</h2><table><thead><tr><th>ID</th><th>Name</th><th>Detail</th></tr></thead><tbody>{gate_rows}</tbody></table></section>
<section><h2>Scenarios</h2><table><thead><tr><th>ID</th><th>Name</th><th>Result</th><th>Duration</th><th>Detail</th><th>Screenshot</th></tr></thead><tbody>{scenario_rows}</tbody></table></section>
<section><h2>Failures / stacktraces</h2><ul>{fail_block}</ul></section>
<section><h2>Log excerpt</h2><pre>{_esc(chr(10).join(payload.get('log_excerpt') or []))}</pre></section>
</main></body></html>"""
    path.write_text(doc, encoding="utf-8")
    return path


def _write_pdf(path: Path, payload: Dict[str, Any]) -> Path:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception:
        path.write_text("reportlab unavailable — see release_report.html/json\n", encoding="utf-8")
        return path

    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    y = height - 54
    c.setFont("Helvetica-Bold", 16)
    c.drawString(54, y, payload["title"][:80])
    y -= 22
    c.setFont("Helvetica-Bold", 14)
    c.drawString(54, y, f"VERDICT: {payload['verdict']}")
    y -= 18
    c.setFont("Helvetica", 10)
    c.drawString(54, y, f"{payload.get('generated_at')} | {payload.get('platform')}")
    y -= 28
    c.setFont("Helvetica-Bold", 12)
    c.drawString(54, y, "Scenario results")
    y -= 16
    c.setFont("Helvetica", 9)
    for s in payload.get("scenarios", []):
        if y < 72:
            c.showPage()
            y = height - 54
            c.setFont("Helvetica", 9)
        line = f"{s.get('id')} [{'PASS' if s.get('ok') else 'FAIL'}] {s.get('name')} — {str(s.get('detail') or s.get('error') or '')[:90]}"
        c.drawString(54, y, line[:110])
        y -= 12
    y -= 12
    if y < 120:
        c.showPage()
        y = height - 54
    c.setFont("Helvetica-Bold", 12)
    c.drawString(54, y, "Failed scenarios")
    y -= 16
    c.setFont("Helvetica", 9)
    fails = payload.get("failed_scenarios") or []
    if not fails:
        c.drawString(54, y, "None")
    for f in fails:
        for chunk in (f.get("stacktrace") or f.get("error") or f.get("detail") or "").splitlines()[:20]:
            if y < 72:
                c.showPage()
                y = height - 54
                c.setFont("Helvetica", 9)
            c.drawString(54, y, chunk[:110])
            y -= 11
        y -= 8
    c.save()
    return path
