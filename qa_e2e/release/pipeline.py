"""
Ordered release validation pipeline.

1 Launch → 2 Environment → 3 AI startup → 4 Index catalog → 5 Scenarios S01–S30
Overall PASS only if every gate and every scenario passes.
"""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from qa_e2e.framework.harness import launch_customer_app
from qa_e2e.framework.readiness import probe_readiness
from qa_e2e.framework.ui_driver import UIDriver
from qa_e2e.release.environment import collect_environment, environment_gate_failures
from qa_e2e.release.report_bundle import write_release_reports
from qa_e2e.release.scenario_result import ScenarioResult
from qa_e2e.release.scenarios import SCENARIO_REGISTRY


def _gate(gid: str, name: str, ok: bool, detail: str, **metrics) -> Dict[str, Any]:
    return {"id": gid, "name": name, "ok": ok, "detail": detail, "metrics": metrics}


def run_release_validation(
    *,
    work_dir: Path,
    out_dir: Path,
    scenario_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute the full release validation pipeline.

    Returns the report payload dict. Raises SystemExit semantics via
    ``verdict`` field — callers should exit non-zero on FAIL.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    gates: List[Dict[str, Any]] = []
    scenarios: List[Dict[str, Any]] = []
    session = None
    environment: Dict[str, Any] = {}
    resources: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    logs: List[str] = []
    t0 = time.time()

    try:
        # ── Gate 1: Launch ─────────────────────────────────────────────────
        launch_ok = False
        launch_detail = ""
        try:
            session = launch_customer_app(
                work_dir=work_dir,
                artifact_dir=out_dir,
                human_seed=int(os.environ.get("TILEVISION_QA_SEED", "42")),
                human_speed=float(os.environ.get("TILEVISION_QA_SPEED", "2.0")),
                catalog_tiles=int(os.environ.get("TILEVISION_QA_TILES", "4")),
            )
            launch_ok = bool(session.main_window.isVisible())
            launch_detail = "MainWindow visible, no launch exception"
        except Exception as exc:
            launch_detail = f"{exc.__class__.__name__}: {exc}"
            gates.append(
                _gate("G1", "Launch application", False, launch_detail, stacktrace=traceback.format_exc())
            )
            environment = collect_environment()
            return _finalize(out_dir, environment, gates, scenarios, [], False, t0)

        gates.append(_gate("G1", "Launch application", launch_ok, launch_detail))

        # ── Gate 2: Environment ────────────────────────────────────────────
        environment = collect_environment(session=session)
        env_fails = environment_gate_failures(environment)
        gates.append(
            _gate(
                "G2",
                "Verify environment",
                not env_fails,
                "ok" if not env_fails else "; ".join(env_fails),
                **{
                    k: environment.get(k)
                    for k in (
                        "python",
                        "torch",
                        "torchvision",
                        "faiss",
                        "sqlite",
                        "opencv",
                        "pyside6",
                        "cpu",
                        "ram_gb",
                        "os",
                        "architecture",
                    )
                },
            )
        )

        # ── Gate 3: AI startup ─────────────────────────────────────────────
        ready = probe_readiness(session, require_catalog=False)
        runtime = environment.get("runtime") or {}
        license_ok = bool(runtime.get("license_ok", False))
        cache_ready = Path(session.settings.thumbnail_dir).exists()
        # AppSettings stores paths privately; prove config.json is on disk.
        config_path = Path(getattr(session.settings, "_config_file", ""))
        if not config_path:
            config_path = Path(session.settings.database_path).parents[1] / "config.json"
        config_loaded = config_path.is_file()
        sam2 = environment.get("sam2", {})
        sam2_ready = bool(sam2.get("onnx_encoder") or sam2.get("load_ok"))
        ai_ok = (
            ready.model_loaded
            and ready.faiss_loaded
            and ready.sqlite_connected
            and ready.ui_ready
            and license_ok
            and cache_ready
            and config_loaded
            and sam2_ready
        )
        gates.append(
            _gate(
                "G3",
                "Verify AI startup",
                ai_ok,
                f"model={ready.model_loaded} faiss={ready.faiss_loaded} sqlite={ready.sqlite_connected} "
                f"license={license_ok} cache={cache_ready} config={config_loaded} "
                f"backend={ready.faiss_backend} sam2_ready={sam2_ready} "
                f"config_path={config_path}",
                readiness=asdict(ready) if is_dataclass(ready) else getattr(ready, "__dict__", {}),
                embedding_model_ready=ready.model_loaded,
                dinov2_loaded=ready.model_loaded,
                sam2_loaded=sam2_ready,
                faiss_ready=ready.faiss_loaded,
                sqlite_ready=ready.sqlite_connected,
                cache_ready=cache_ready,
                configuration_loaded=config_loaded,
                license_valid=license_ok,
            )
        )

        driver = UIDriver(session)

        # ── Gate 4: Index catalog ──────────────────────────────────────────
        index_ok = False
        index_detail = ""
        index_metrics: Dict[str, Any] = {}
        try:
            from qa_e2e.framework.indexing import index_catalog_customer_style

            index_t0 = time.time()
            index_metrics = index_catalog_customer_style(session, driver)
            index_metrics["index_time_s"] = round(time.time() - index_t0, 3)
            catalog_images = sum(
                1
                for p in Path(session.catalog_dir).rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
            )
            index_metrics["image_count"] = catalog_images
            index_metrics["embedding_count"] = index_metrics.get("faiss_count")
            index_ok = (
                index_metrics.get("faiss_count", 0) > 0
                and index_metrics.get("sqlite_count", 0) > 0
                and catalog_images > 0
            )
            index_detail = (
                f"images={catalog_images} faiss={index_metrics.get('faiss_count')} "
                f"sqlite={index_metrics.get('sqlite_count')} mode={index_metrics.get('mode')} "
                f"index_time_s={index_metrics.get('index_time_s')}"
            )
            if not index_ok:
                index_detail += " — empty index"
        except Exception as exc:
            index_detail = f"{exc.__class__.__name__}: {exc}"
            index_metrics["stacktrace"] = traceback.format_exc()
        gates.append(_gate("G4", "Index catalog", index_ok, index_detail, **index_metrics))

        # ── Gate 5: Customer scenarios ─────────────────────────────────────
        selected = SCENARIO_REGISTRY
        if scenario_ids:
            wanted = {s.upper() for s in scenario_ids}
            selected = [row for row in SCENARIO_REGISTRY if row[0] in wanted]

        if not (launch_ok and not env_fails and ai_ok and index_ok):
            for sid, name, _fn in selected:
                scenarios.append(
                    ScenarioResult(
                        id=sid,
                        name=name,
                        ok=False,
                        started_at=time.time(),
                        ended_at=time.time(),
                        detail="BLOCKED — prior release gate failed",
                        error="GATE_BLOCKED",
                    ).to_dict()
                )
        else:
            for sid, name, fn in selected:
                action = session.artifacts.begin(f"Release {sid}", detail=name)
                result = fn(session, driver)
                row = result.to_dict()
                session.artifacts.sample_resources()
                if session.artifacts.resources:
                    latest = session.artifacts.resources[-1]
                    row.setdefault("metrics", {})
                    row["metrics"]["rss_mb"] = latest.rss_mb
                    row["metrics"]["cpu_percent"] = latest.cpu_percent
                if not result.ok:
                    bundle = _write_failure_bundle(
                        out_dir=out_dir,
                        scenario=row,
                        environment=environment,
                        recent_actions=session.artifacts.actions[-12:],
                        logs=session.logs.messages()[-200:],
                    )
                    row["failure_bundle"] = str(bundle)
                scenarios.append(row)
                session.artifacts.end(
                    action,
                    ok=result.ok,
                    detail=result.detail or result.error,
                    screenshot_widget=session.main_window if not result.screenshot else None,
                    metrics=result.metrics,
                )
                # Incremental artifacts so a hard crash still leaves inspectable evidence.
                try:
                    interim_pass = all(g.get("ok") for g in gates) and all(
                        s.get("ok") for s in scenarios
                    )
                    _finalize(
                        out_dir,
                        environment,
                        gates,
                        scenarios,
                        session.logs.messages()[-300:],
                        interim_pass,
                        t0,
                        resources=[asdict(r) for r in session.artifacts.resources],
                        actions=[asdict(a) for a in session.artifacts.actions],
                    )
                except Exception:
                    pass

        logs = session.logs.messages()[-500:]
        resources = [asdict(r) for r in session.artifacts.resources]
        actions = [asdict(a) for a in session.artifacts.actions]
        try:
            session.artifacts.dump_json()
        except Exception:
            pass
        try:
            (out_dir / "tilevision_release.log").write_text("\n".join(logs), encoding="utf-8")
        except Exception:
            pass
        # Also copy the rotating app log from the isolated HOME if present.
        try:
            from src.utils.logger import get_log_file_path
            import shutil

            app_log = get_log_file_path("tilevision.log")
            if app_log.is_file():
                shutil.copy2(app_log, out_dir / "tilevision_app.log")
        except Exception:
            pass

        all_scenarios_pass = bool(scenarios) and all(s.get("ok") for s in scenarios)
        all_gates_pass = all(g.get("ok") for g in gates)
        final_pass = all_gates_pass and all_scenarios_pass
    except Exception as exc:
        gates.append(
            _gate(
                "GX",
                "Pipeline crash",
                False,
                f"{exc.__class__.__name__}: {exc}",
                stacktrace=traceback.format_exc(),
            )
        )
        if not environment:
            environment = collect_environment(session=session)
        final_pass = False
        if session is not None:
            try:
                logs = session.logs.messages()[-500:]
            except Exception:
                pass
    finally:
        try:
            if session is not None:
                session.close()
        except Exception:
            pass

    return _finalize(
        out_dir,
        environment,
        gates,
        scenarios,
        logs,
        final_pass,
        t0,
        resources=resources,
        actions=actions,
    )


def _write_failure_bundle(
    *,
    out_dir: Path,
    scenario: Dict[str, Any],
    environment: Dict[str, Any],
    recent_actions: list,
    logs: List[str],
) -> Path:
    import json

    fail_dir = Path(out_dir) / "failures" / str(scenario.get("id") or "unknown")
    fail_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": scenario,
        "environment": environment,
        "recent_actions": [
            {
                "name": a.name,
                "ok": a.ok,
                "detail": a.detail,
                "duration_s": a.duration_s,
                "screenshot": a.screenshot,
                "metrics": a.metrics,
            }
            for a in recent_actions
        ],
        "logs": logs,
        "stacktrace": scenario.get("stacktrace") or scenario.get("error") or "",
        "screenshot": scenario.get("screenshot") or "",
    }
    path = fail_dir / "failure.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (fail_dir / "stacktrace.txt").write_text(payload["stacktrace"], encoding="utf-8")
    (fail_dir / "logs.txt").write_text("\n".join(logs), encoding="utf-8")
    return path


def _finalize(
    out_dir,
    environment,
    gates,
    scenarios,
    logs,
    final_pass,
    t0,
    *,
    resources=None,
    actions=None,
) -> Dict[str, Any]:
    paths = write_release_reports(
        out_dir=out_dir,
        environment=environment,
        gates=gates,
        scenarios=scenarios,
        logs=logs,
        final_pass=final_pass,
        resources=resources or [],
        actions=actions or [],
    )
    payload = {
        "verdict": "PASS" if final_pass else "FAIL",
        "duration_s": time.time() - t0,
        "gates_passed": sum(1 for g in gates if g.get("ok")),
        "gates_total": len(gates),
        "scenarios_passed": sum(1 for s in scenarios if s.get("ok")),
        "scenarios_total": len(scenarios),
        "reports": {k: str(v) for k, v in paths.items()},
        "environment": environment,
        "gates": gates,
        "scenarios": scenarios,
    }
    (Path(out_dir) / "release_summary.json").write_text(
        __import__("json").dumps(
            {
                k: payload[k]
                for k in (
                    "verdict",
                    "duration_s",
                    "gates_passed",
                    "gates_total",
                    "scenarios_passed",
                    "scenarios_total",
                    "reports",
                )
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return payload
