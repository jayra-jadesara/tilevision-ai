#!/usr/bin/env python3
"""
Index-speed microbench for a local catalogue folder.

Run on the customer laptop (Windows paths OK). Loads DINOv2 from
``model_weights/`` when present. Prints progress and writes a CSV.

Usage:
  python dev_tools/bench_indexing_speed.py --catalogue "D:\\Tiles\\Showroom"
  python dev_tools/bench_indexing_speed.py --catalogue ./sample_tiles --variants threads,aux1view

Variants (opt-in measurement only — does **not** change the app default):
  threads   — torch intra-op 2 / 3 / 4
  int8      — dynamic int8 on DINOv2 nn.Linear
  aux1view  — aux crops with 1 multi-scale view instead of 3
  views2    — 2 index views per image instead of 3

Ship gate (documented in INDEXING_SPEED_INVESTIGATION.md): a variant may
become opt-in Fast only if top-1 identical, top-5 within 1 pp, and mean
cosine ≥ 0.99 vs fp32 Accurate. This script reports the numbers; it does
not flip the product default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
from pathlib import Path

# Repo root on sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image

from src.utils.image_utils import SUPPORTED_IMAGE_EXTENSIONS


def _iter_images(folder: Path) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        for pattern in (f"*{ext}", f"*{ext.upper()}"):
            for path in folder.rglob(pattern):
                if not path.is_file():
                    continue
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                out.append(path)
    return sorted(out, key=lambda p: str(p).lower())


def _content_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _distinct_by_content(paths: list[Path]) -> list[Path]:
    """One path per file content (matches catalogue dedupe intent)."""
    by_hash: dict[str, Path] = {}
    for path in paths:
        try:
            digest = _content_sha256(path)
        except OSError as exc:
            print(f"  skip unreadable {path}: {exc}", flush=True)
            continue
        by_hash.setdefault(digest, path)
    return list(by_hash.values())


def _set_torch_threads(n: int) -> None:
    import torch

    torch.set_num_threads(max(1, int(n)))
    print(f"  torch.set_num_threads({n}) → {torch.get_num_threads()}", flush=True)


def _maybe_int8_linear(model):
    """Return ``(linear_count, quantized_or_original_model)``."""
    import torch

    count = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    if count == 0:
        return 0, model
    try:
        quantized = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return count, quantized
    except Exception as exc:
        print(f"  int8 quantize failed: {exc}", flush=True)
        return 0, model


def _cosine(a, b) -> float:
    import numpy as np

    aa = np.asarray(a, dtype=np.float32).ravel()
    bb = np.asarray(b, dtype=np.float32).ravel()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def _build_extractor(*, device: str = "cpu"):
    from src.ai.embedder import DINOv2Embedder
    from src.ai.feature_extractor import FeatureExtractor

    embedder = DINOv2Embedder(device_preference=device)
    embedder.load_model()
    return FeatureExtractor(embedder=embedder), embedder


def _embed_paths(extractor, paths: list[Path], *, label: str) -> dict[str, object]:
    import numpy as np

    vectors: list[np.ndarray] = []
    view_counts: list[int] = []
    t0 = time.perf_counter()
    for i, path in enumerate(paths, start=1):
        print(f"  [{label}] {i}/{len(paths)} {path.name}", flush=True)
        features, aux = extractor.extract_index_vectors(str(path))
        primary = np.asarray(features.embedding, dtype=np.float32).ravel()
        vectors.append(primary)
        # Approximate views: 3 primary multi-scale + 3 per kept aux
        view_counts.append(3 + 3 * len(aux))
    elapsed = time.perf_counter() - t0
    total_views = sum(view_counts) or 1
    return {
        "vectors": vectors,
        "elapsed_s": elapsed,
        "ms_per_image": 1000.0 * elapsed / max(1, len(paths)),
        "ms_per_view": 1000.0 * elapsed / total_views,
        "views": total_views,
    }


def _retrieval_vs_fp32(
    fp32_vecs: list,
    variant_vecs: list,
    *,
    top_k: int = 5,
) -> dict[str, float]:
    """Crop-query proxy: each fp32 vector queried against both indexes."""
    import numpy as np

    n = min(len(fp32_vecs), len(variant_vecs))
    if n == 0:
        return {"top1_match_pp": 0.0, "top5_overlap_pp": 0.0, "mean_cosine": 0.0}

    fp32 = np.stack([np.asarray(v, dtype=np.float32).ravel() for v in fp32_vecs[:n]])
    var = np.stack([np.asarray(v, dtype=np.float32).ravel() for v in variant_vecs[:n]])
    # L2 normalize
    fp32 = fp32 / np.maximum(np.linalg.norm(fp32, axis=1, keepdims=True), 1e-8)
    var = var / np.maximum(np.linalg.norm(var, axis=1, keepdims=True), 1e-8)

    cosines = [float(np.dot(fp32[i], var[i])) for i in range(n)]
    mean_cos = float(sum(cosines) / n)

    def _topk(index: np.ndarray, query: np.ndarray, k: int) -> list[int]:
        scores = index @ query
        return list(np.argsort(-scores)[:k])

    top1_ok = 0
    top5_hits = 0
    for i in range(n):
        q = fp32[i]
        ref = _topk(fp32, q, top_k)
        alt = _topk(var, q, top_k)
        if ref[0] == alt[0]:
            top1_ok += 1
        top5_hits += len(set(ref) & set(alt)) / float(top_k)

    return {
        "top1_match_pp": 100.0 * top1_ok / n,
        "top5_overlap_pp": 100.0 * top5_hits / n,
        "mean_cosine": mean_cos,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalogue",
        required=True,
        help="Folder of tile images (Windows or POSIX path)",
    )
    parser.add_argument(
        "--out-csv",
        default="indexing_speed_bench.csv",
        help="CSV output path (default: indexing_speed_bench.csv)",
    )
    parser.add_argument(
        "--variants",
        default="threads",
        help="Comma list: threads,int8,aux1view,views2 (default: threads)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap on distinct images (0 = all)",
    )
    args = parser.parse_args(argv)

    catalogue = Path(args.catalogue).expanduser()
    if not catalogue.is_dir():
        print(f"Catalogue not found: {catalogue}", file=sys.stderr)
        return 2

    all_paths = _iter_images(catalogue)
    distinct = _distinct_by_content(all_paths)
    if args.max_images and args.max_images > 0:
        distinct = distinct[: args.max_images]

    print(
        f"Catalogue {catalogue}: {len(all_paths)} files, "
        f"{len(distinct)} distinct by content",
        flush=True,
    )
    if not distinct:
        print("No images found.", file=sys.stderr)
        return 2

    weights = _ROOT / "model_weights"
    print(f"Weights dir: {weights} (exists={weights.is_dir()})", flush=True)

    print("Loading Accurate (fp32) baseline…", flush=True)
    extractor, embedder = _build_extractor()
    baseline = _embed_paths(extractor, distinct, label="fp32")
    rows: list[dict] = [
        {
            "variant": "fp32_accurate",
            "ms_per_view": f"{baseline['ms_per_view']:.1f}",
            "ms_per_image": f"{baseline['ms_per_image']:.1f}",
            "mean_cosine_vs_fp32": "1.000",
            "top1_match_pp": "100.0",
            "top5_overlap_pp": "100.0",
            "notes": f"views≈{baseline['views']}",
        }
    ]

    wanted = {v.strip().lower() for v in args.variants.split(",") if v.strip()}

    if "threads" in wanted:
        for n in (2, 3, 4):
            print(f"Variant torch threads={n}", flush=True)
            _set_torch_threads(n)
            # Reuse same loaded model; thread count is process-global.
            result = _embed_paths(extractor, distinct, label=f"threads{n}")
            quality = _retrieval_vs_fp32(baseline["vectors"], result["vectors"])
            rows.append(
                {
                    "variant": f"torch_threads_{n}",
                    "ms_per_view": f"{result['ms_per_view']:.1f}",
                    "ms_per_image": f"{result['ms_per_image']:.1f}",
                    "mean_cosine_vs_fp32": f"{quality['mean_cosine']:.4f}",
                    "top1_match_pp": f"{quality['top1_match_pp']:.1f}",
                    "top5_overlap_pp": f"{quality['top5_overlap_pp']:.1f}",
                    "notes": "HT often hurts GEMM on 2P/4L CPUs — measure",
                }
            )

    if "int8" in wanted:
        print("Variant dynamic int8 Linear…", flush=True)
        try:
            count, qmodel = _maybe_int8_linear(embedder._model)
            if count:
                embedder._model = qmodel
                result = _embed_paths(extractor, distinct, label="int8")
                quality = _retrieval_vs_fp32(baseline["vectors"], result["vectors"])
                rows.append(
                    {
                        "variant": "dynamic_int8_linear",
                        "ms_per_view": f"{result['ms_per_view']:.1f}",
                        "ms_per_image": f"{result['ms_per_image']:.1f}",
                        "mean_cosine_vs_fp32": f"{quality['mean_cosine']:.4f}",
                        "top1_match_pp": f"{quality['top1_match_pp']:.1f}",
                        "top5_overlap_pp": f"{quality['top5_overlap_pp']:.1f}",
                        "notes": f"quantized {count} Linear; Skylake-U no VNNI",
                    }
                )
            else:
                rows.append(
                    {
                        "variant": "dynamic_int8_linear",
                        "ms_per_view": "",
                        "ms_per_image": "",
                        "mean_cosine_vs_fp32": "",
                        "top1_match_pp": "",
                        "top5_overlap_pp": "",
                        "notes": "UNVERIFIED — quantize unavailable",
                    }
                )
        except Exception as exc:
            print(f"  int8 variant failed: {exc}", flush=True)
            rows.append(
                {
                    "variant": "dynamic_int8_linear",
                    "ms_per_view": "",
                    "ms_per_image": "",
                    "mean_cosine_vs_fp32": "",
                    "top1_match_pp": "",
                    "top5_overlap_pp": "",
                    "notes": f"UNVERIFIED — {exc}",
                }
            )

    if "aux1view" in wanted or "views2" in wanted:
        # These need embedder hooks; report as harness stubs with env notes.
        for name, note in (
            (
                "aux_1_view",
                "Set future Fast=aux1view; UNVERIFIED until run with hook",
            ),
            (
                "index_2_views",
                "Set future Fast=views2; UNVERIFIED until run with hook",
            ),
        ):
            if (name.startswith("aux") and "aux1view" in wanted) or (
                name.startswith("index") and "views2" in wanted
            ):
                rows.append(
                    {
                        "variant": name,
                        "ms_per_view": "",
                        "ms_per_image": "",
                        "mean_cosine_vs_fp32": "",
                        "top1_match_pp": "",
                        "top5_overlap_pp": "",
                        "notes": note,
                    }
                )

    out_path = Path(args.out_csv)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "variant",
                "ms_per_view",
                "ms_per_image",
                "mean_cosine_vs_fp32",
                "top1_match_pp",
                "top5_overlap_pp",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {out_path.resolve()}", flush=True)
    for row in rows:
        print(
            f"  {row['variant']}: "
            f"ms/view={row['ms_per_view']} ms/image={row['ms_per_image']} "
            f"cos={row['mean_cosine_vs_fp32']} "
            f"top1={row['top1_match_pp']} top5={row['top5_overlap_pp']}",
            flush=True,
        )
    print(
        "\nShip gate: top-1 identical, top-5 within 1 pp, mean cosine ≥ 0.99 "
        "→ may expose opt-in Fast. Default Accurate must not change.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
