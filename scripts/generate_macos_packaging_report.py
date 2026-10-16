#!/usr/bin/env python3
"""Generate packaging / dependency / PyInstaller reports for a macOS .app + DMG."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from version import APP_VERSION  # noqa: E402


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"(unavailable: {exc})"


def _du_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        out = subprocess.check_output(["du", "-sm", str(path)], text=True)
        return float(out.split()[0])
    except Exception:
        return None


def _find(app: Path, pattern: str) -> list[str]:
    hits = list(app.glob(pattern))
    return [str(p.relative_to(app)) for p in hits[:20]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True, type=Path)
    parser.add_argument("--dmg", default="", type=Path)
    parser.add_argument("--arch", required=True, choices=["x86_64", "arm64"])
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    app: Path = args.app
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    exe = app / "Contents" / "MacOS" / "TileVisionAI"
    codesign = _run(["codesign", "-dv", "--verbose=4", str(app)]) if app.exists() else "n/a"
    unsigned = "Unsigned build"
    if "code object is not signed" in codesign.lower() or "not signed" in codesign.lower():
        unsigned = "Unsigned build"
    elif app.exists() and "Signature=" in codesign and "adhoc" not in codesign.lower():
        unsigned = "Signed (verify Gatekeeper separately)"
    else:
        unsigned = "Unsigned build"

    deps = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "arch_target": args.arch,
    }
    for mod in (
        "torch",
        "torchvision",
        "faiss",
        "cv2",
        "PIL",
        "reportlab",
        "onnxruntime",
        "PySide6",
        "numpy",
        "transformers",
    ):
        try:
            m = __import__(mod if mod != "PIL" else "PIL")
            if mod == "PIL":
                import PIL

                deps[mod] = getattr(PIL, "__version__", "present")
            else:
                deps[mod] = getattr(m, "__version__", "present")
        except Exception as exc:
            deps[mod] = f"unavailable ({exc.__class__.__name__})"

    packaging = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "app_version": APP_VERSION,
        "app_path": str(app),
        "app_exists": app.is_dir(),
        "executable": str(exe),
        "executable_exists": exe.is_file(),
        "file_exe": _run(["file", str(exe)]) if exe.is_file() else "",
        "app_size_mb": _du_mb(app),
        "dmg_path": str(args.dmg) if args.dmg else "",
        "dmg_exists": bool(args.dmg) and Path(args.dmg).is_file(),
        "dmg_size_mb": _du_mb(Path(args.dmg)) if args.dmg else None,
        "signing_status": unsigned,
        "codesign_raw": codesign[:2000],
        "universal2": False,
        "universal2_reason": (
            "Intel Mac builds pin torch==2.2.2 (last x86_64 macOS wheels). "
            "Apple Silicon builds use current torch from requirements.txt. "
            "Those native extensions cannot be combined with lipo into a "
            "reliable Universal2 .app. Ship TileVision-AI-Intel.dmg and "
            "TileVision-AI-AppleSilicon.dmg separately."
        ),
        "bundled_models": {
            "dinov2": _find(app, "**/model_weights/dinov2-large/config.json"),
            "sam2_onnx": _find(app, "**/model_weights/sam2.1-hiera-tiny-onnx/*.encoder.onnx"),
        },
        "resources": _find(app, "**/src/resources/**"),
        "env": {
            "TILEVISION_BUNDLE_SAM2": os.environ.get("TILEVISION_BUNDLE_SAM2", ""),
            "TILEVISION_OFFLINE_MODEL": os.environ.get("TILEVISION_OFFLINE_MODEL", ""),
            "MACOS_BUILD_ARCH": os.environ.get("MACOS_BUILD_ARCH", ""),
        },
        "macos_paths_note": {
            "settings_sqlite_faiss_logs": "~/.tilevision_ai/ (legacy home-dot dir used by AppSettings)",
            "license_crypto_store": "~/Library/Application Support/TileVisionAI/.lic/",
            "pdf_export": "user-chosen path (defaults under home)",
            "windows_paths": "not used on darwin",
        },
    }

    pyinstaller = {
        "spec": "packaging/tilevision_mac.spec",
        "bundle_name": "TileVision AI.app",
        "bundle_identifier": "com.jdsoftware.tilevisionai",
        "onefolder": True,
        "console": False,
        "collect_all": ["torch", "torchvision", "onnxruntime", "cv2", "faiss", "reportlab", "PySide6"],
        "models_bundled_offline": True,
    }

    (out / "packaging_report.json").write_text(json.dumps(packaging, indent=2), encoding="utf-8")
    (out / "dependency_report.json").write_text(json.dumps(deps, indent=2), encoding="utf-8")
    (out / "pyinstaller_report.json").write_text(json.dumps(pyinstaller, indent=2), encoding="utf-8")

    md = [
        f"# TileVision AI macOS Packaging Report",
        "",
        f"- Generated: `{packaging['generated_at']}`",
        f"- App version: `{APP_VERSION}`",
        f"- Target arch: `{args.arch}`",
        f"- App: `{app}` ({packaging['app_size_mb']} MB)",
        f"- DMG: `{args.dmg}` ({packaging['dmg_size_mb']} MB)",
        f"- Signing: **{unsigned}**",
        f"- Universal2: **No** — {packaging['universal2_reason']}",
        "",
        "## Bundled models",
        f"- DINOv2: `{packaging['bundled_models']['dinov2']}`",
        f"- SAM2 ONNX: `{packaging['bundled_models']['sam2_onnx']}`",
        "",
        "## Dependencies (build environment)",
    ]
    for k, v in deps.items():
        md.append(f"- {k}: `{v}`")
    md += [
        "",
        "## Known limitations",
        "- Unsigned build — customers must Right-click → Open on first launch until notarized.",
        "- No Universal2 binary (torch Intel pin vs Silicon current).",
        "- Full Release Validation Suite (30/30) must be run on a Mac host from the unpacked .app.",
        "",
    ]
    (out / "packaging_report.md").write_text("\n".join(md), encoding="utf-8")
    (out / "known_limitations.md").write_text(
        "\n".join(
            [
                "# Known limitations — macOS release",
                "",
                "1. **Unsigned build** — Apple Developer credentials were not available in CI; Gatekeeper requires Right-click → Open.",
                "2. **No Universal2** — Intel torch 2.2.2 vs Apple Silicon current torch cannot be lipo'd.",
                "3. **Dual DMGs required** — `TileVision-AI-Intel.dmg` and `TileVision-AI-AppleSilicon.dmg`.",
                "4. **Data directory** — primary app data remains `~/.tilevision_ai` (SQLite, FAISS, logs, config).",
                "5. **License crypto store** — `~/Library/Application Support/TileVisionAI/.lic/`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote reports under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
