"""Generate a complete HTML QA report for human review."""

from __future__ import annotations

import html
import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from qa_e2e.framework.collectors import ArtifactCollector


def _esc(text: Any) -> str:
    return html.escape("" if text is None else str(text))


def write_html_report(
    *,
    out_dir: Path,
    collector: ArtifactCollector,
    readiness: Optional[Dict[str, Any]] = None,
    expectations: Optional[List[Dict[str, Any]]] = None,
    log_excerpt: Optional[List[str]] = None,
    final_pass: bool,
    title: str = "TileVision AI — Human E2E QA Report",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    collector.dump_json()

    verdict = "PASS" if final_pass else "FAIL"
    color = "#0a7a32" if final_pass else "#b00020"
    actions_html = []
    for a in collector.actions:
        shot = (
            f'<div class="shot"><img src="{_esc(a.screenshot)}" alt="screenshot"/></div>'
            if a.screenshot
            else ""
        )
        badge = "ok" if a.ok else "bad"
        actions_html.append(
            f"""
            <article class="action {badge}">
              <h3>{_esc(a.name)} <span class="badge">{badge.upper()}</span></h3>
              <p class="meta">{a.duration_s:.2f}s — {_esc(a.detail)}</p>
              <pre>{_esc(json.dumps(a.metrics, indent=2))}</pre>
              {shot}
            </article>
            """
        )

    fail_html = "".join(
        f"<li><strong>{_esc(f.get('action'))}</strong>: {_esc(f.get('detail'))}</li>"
        for f in collector.failures
    ) or "<li>None</li>"

    exp_html = []
    for e in expectations or []:
        cls = "ok" if e.get("ok") else "bad"
        exp_html.append(
            f"<tr class='{cls}'><td>{_esc(e.get('query_id'))}</td>"
            f"<td>{_esc(e.get('expected_product'))}</td>"
            f"<td>{_esc(e.get('top_product'))}</td>"
            f"<td>{_esc(e.get('rank_of_expected'))}</td>"
            f"<td>{_esc(e.get('detail'))}</td></tr>"
        )

    ready_pre = _esc(json.dumps(readiness or {}, indent=2))
    logs_pre = _esc("\n".join(log_excerpt or collector.notes))
    resources = collector.resources[-1] if collector.resources else None
    res_line = (
        f"RSS {resources.rss_mb:.1f} MiB · CPU {resources.cpu_percent:.1f}%"
        if resources
        else "n/a"
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{_esc(title)} — {verdict}</title>
  <style>
    :root {{
      --ink: #1c1917;
      --muted: #57534e;
      --bg: #f5f2eb;
      --card: #fffdf8;
      --line: #e7e0d5;
      --ok: #0a7a32;
      --bad: #b00020;
    }}
    body {{
      margin: 0; font-family: "Iowan Old Style", "Palatino Linotype", Palatino, serif;
      color: var(--ink); background:
        radial-gradient(ellipse at 10% 0%, #efe6d6 0%, transparent 45%),
        radial-gradient(ellipse at 90% 10%, #d9e2ea 0%, transparent 40%),
        var(--bg);
    }}
    header {{
      padding: 48px 56px 24px; border-bottom: 1px solid var(--line);
    }}
    header h1 {{ margin: 0 0 8px; font-size: 40px; letter-spacing: -0.03em; }}
    .verdict {{
      display: inline-block; padding: 6px 14px; border-radius: 999px;
      color: white; background: {color}; font-weight: 700; letter-spacing: 0.08em;
    }}
    main {{ padding: 32px 56px 80px; max-width: 1100px; }}
    section {{ margin: 36px 0; }}
    h2 {{ font-size: 22px; margin: 0 0 12px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .card {{
      background: var(--card); border: 1px solid var(--line); border-radius: 12px;
      padding: 16px 18px;
    }}
    .action {{
      background: var(--card); border: 1px solid var(--line); border-radius: 12px;
      padding: 16px 18px; margin: 0 0 14px;
    }}
    .action.bad {{ border-color: #f0b4b4; }}
    .badge {{
      font-size: 11px; vertical-align: middle; margin-left: 8px;
      padding: 2px 8px; border-radius: 999px; background: #e7e0d5;
    }}
    .action.ok .badge {{ background: #d7f0df; color: var(--ok); }}
    .action.bad .badge {{ background: #f8d6d6; color: var(--bad); }}
    .meta {{ color: var(--muted); margin-top: 0; }}
    pre {{
      white-space: pre-wrap; background: #f8f4ec; padding: 12px; border-radius: 8px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
    }}
    img {{ max-width: 100%; border-radius: 8px; border: 1px solid var(--line); }}
    table {{ width: 100%; border-collapse: collapse; background: var(--card); }}
    th, td {{ border: 1px solid var(--line); padding: 8px 10px; text-align: left; font-size: 14px; }}
    tr.bad td {{ background: #fff1f1; }}
    tr.ok td {{ background: #f3faf5; }}
    ul {{ line-height: 1.5; }}
  </style>
</head>
<body>
  <header>
    <div class="verdict">{verdict}</div>
    <h1>{_esc(title)}</h1>
    <p class="meta">
      Generated {time.strftime("%Y-%m-%d %H:%M:%S")} ·
      {_esc(platform.platform())} · {_esc(platform.machine())} ·
      Resources: {_esc(res_line)}
    </p>
  </header>
  <main>
    <section class="grid">
      <div class="card">
        <h2>Readiness</h2>
        <pre>{ready_pre}</pre>
      </div>
      <div class="card">
        <h2>Failures</h2>
        <ul>{fail_html}</ul>
        <h2>Notes</h2>
        <pre>{_esc(chr(10).join(collector.notes))}</pre>
      </div>
    </section>

    <section>
      <h2>Customer actions</h2>
      {''.join(actions_html) or '<p>No actions recorded.</p>'}
    </section>

    <section>
      <h2>Expected dataset comparison</h2>
      <table>
        <thead><tr>
          <th>Query</th><th>Expected</th><th>Top</th><th>Rank</th><th>Detail</th>
        </tr></thead>
        <tbody>{''.join(exp_html) or '<tr><td colspan="5">No expectation rows</td></tr>'}</tbody>
      </table>
    </section>

    <section>
      <h2>Log excerpt / search diagnostics</h2>
      <pre>{logs_pre}</pre>
    </section>
  </main>
</body>
</html>
"""
    path = out_dir / "qa_report.html"
    path.write_text(doc, encoding="utf-8")
    summary = {
        "verdict": verdict,
        "actions": len(collector.actions),
        "failures": len(collector.failures),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
