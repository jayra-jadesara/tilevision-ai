#!/usr/bin/env python3
"""
Real Apple Silicon MPS validation helper for TileVision AI.

CI cannot prove hang-safety. Run this on a native arm64 Mac (M1/M2/M3/…)
before claiming Mac Silicon search parity.

Usage (from repo root, after ``source scripts/install_mac_deps.sh arm64``):

  "$MACOS_PYTHON_PATH" scripts/validate_mac_silicon_mps.py
  "$MACOS_PYTHON_PATH" scripts/validate_mac_silicon_mps.py --images /path/to/photos --rounds 30

What it does
------------
1. Confirms the process is Apple Silicon + MPS available.
2. Loads DINOv2 on MPS and runs timed query embeds (default 30 rounds).
3. Prints per-query timings and a summary (min / median / max).
4. Prints the hang-watchdog timeout in force and the exact log lines to
   watch for if a forward stalls.

GUI batch (founder checklist — 20–30 searches)
----------------------------------------------
1. Install / run this branch's Apple Silicon build (or ``python main.py``
   from the arm64 venv above).
2. Index a small catalogue if needed (or use an existing one built on this
   same Mac — do not mix with an Intel-CPU index if you care about ranks).
3. Run at least 20–30 searches mixing:
     - plain drag-and-drop
     - Auto Crop
     - Precise Crop & Search
4. Tail the app log (Help → Open Logs, or ``~/Library/Logs/TileVisionAI/`` /
   the path printed at startup) while searching.

What a healthy run looks like
-----------------------------
- Lines like:
    DINOv2 query forward on MPS with hang watchdog (timeout=20.0s views=…)
- Query timings stay in a tight band (typically well under a few seconds per
  forward once the model is warm). This script prints concrete numbers for
  your machine — use them to decide whether to tighten
  TILEVISION_MPS_QUERY_TIMEOUT_S.

What a stalled / hung MPS forward looks like
--------------------------------------------
- The watchdog line above appears, then ~20s later (default timeout):
    MPS query forward timed out — switching DINOv2 to CPU so search continues
    (hang watchdog). (…)
- After that, further searches log CPU inference (no more MPS watchdog lines).
- A query that sits with no result and no timeout line for far longer than
  the watchdog budget means the watchdog did not fire — report that as a bug.

Exit codes: 0 = measured OK on MPS; 2 = not Apple Silicon / no MPS;
            1 = embed failure.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _require_apple_silicon_mps() -> None:
    from src.utils.platform_info import is_apple_silicon, mac_machine
    import torch

    if not is_apple_silicon():
        print(
            f"ERROR: This script must run on native Apple Silicon "
            f"(machine={mac_machine()!r}). Refusing to continue.",
            file=sys.stderr,
        )
        sys.exit(2)
    mps = getattr(torch.backends, "mps", None)
    if not (mps and mps.is_available()):
        print("ERROR: torch.backends.mps is not available.", file=sys.stderr)
        sys.exit(2)


def _load_images(path: Path | None, rounds: int):
    from PIL import Image

    images = []
    if path is not None:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
        files = sorted(
            p for p in path.rglob("*") if p.suffix.lower() in exts
        )
        for file_path in files:
            try:
                images.append(Image.open(file_path).convert("RGB"))
            except Exception as exc:
                print(f"skip {file_path}: {exc}")
        if not images:
            print(f"ERROR: no images under {path}", file=sys.stderr)
            sys.exit(1)
    else:
        # Synthetic tiles covering a few sizes (no customer photos required).
        for i in range(max(rounds, 8)):
            size = 256 + (i % 5) * 64
            images.append(Image.new("RGB", (size, size), color=(40 + i * 3, 80, 120)))
    return images


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Optional folder of real photos (otherwise synthetic RGB tiles).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=30,
        help="Number of query embeds to time (default 30).",
    )
    args = parser.parse_args()

    _require_apple_silicon_mps()

    from src.ai.embedder import (
        DINOv2Embedder,
        _DEFAULT_MPS_QUERY_TIMEOUT_S,
        mps_query_forward_timeout_s,
    )
    from src.ai.gpu_info import configure_mps_fallback

    configure_mps_fallback()
    timeout_s = mps_query_forward_timeout_s()
    print(f"MPS query hang-watchdog timeout: {timeout_s:.1f}s "
          f"(default {_DEFAULT_MPS_QUERY_TIMEOUT_S:.1f}s; "
          f"override with TILEVISION_MPS_QUERY_TIMEOUT_S)")
    print("Watch log for: 'DINOv2 query forward on MPS with hang watchdog'")
    print("Hang signal:   'MPS query forward timed out — switching DINOv2 to CPU'")

    images = _load_images(args.images, args.rounds)
    embedder = DINOv2Embedder(device_preference="auto")
    embedder.load_model()
    if embedder._device.type != "mps":
        print(
            f"ERROR: embedder landed on {embedder._device.type}, expected mps",
            file=sys.stderr,
        )
        return 2

    samples_ms: list[float] = []
    n = min(args.rounds, max(len(images) * 4, args.rounds))
    for i in range(n):
        image = images[i % len(images)]
        t0 = time.perf_counter()
        vec = embedder._extract_batch([image], for_query=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        samples_ms.append(elapsed_ms)
        device = embedder._device.type
        print(
            f"query[{i+1:02d}/{n}] {elapsed_ms:8.1f} ms  "
            f"device={device}  dim={getattr(vec, 'shape', None)}  "
            f"fallback_done={embedder._mps_cpu_fallback_done}"
        )
        if embedder._mps_cpu_fallback_done and device == "cpu":
            print(
                "WATCHDOG FIRED — MPS hang/timeout fallback engaged. "
                "Collect this log and stop the GUI batch for review."
            )
            break

    if not samples_ms:
        return 1

    ordered = sorted(samples_ms)
    median = statistics.median(ordered)
    print(
        f"\nSummary over {len(samples_ms)} queries: "
        f"min={ordered[0]:.1f} ms  median={median:.1f} ms  "
        f"max={ordered[-1]:.1f} ms  device_final={embedder._device.type}"
    )
    if embedder._device.type == "mps" and not embedder._mps_cpu_fallback_done:
        suggested = max(5.0, (median / 1000.0) * 10.0)
        print(
            f"Suggested TILEVISION_MPS_QUERY_TIMEOUT_S ≈ {suggested:.1f} "
            f"(~10× measured median) if you want a tighter hang budget than "
            f"the default { _DEFAULT_MPS_QUERY_TIMEOUT_S:.1f}s."
        )
        print("MPS query path stayed on Metal for the whole batch — OK.")
        return 0

    print("MPS did not stay active for the full batch — review watchdog logs.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
