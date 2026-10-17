#!/usr/bin/env python3
"""
Standalone description-search eval on a real image catalogue.

Loads CLIP from model_weights/ (no app DB), embeds each unique image once
(sha256 dedupe), labels colours from pixels, and reports precision@5/@10 for:

  - raw single prompt
  - prompt templates (ensemble)
  - templates + relevance cutoff
  - templates + cutoff + colour-word RRF fusion

Also prints distribution-aware display % stats per query.

Precision tables are UNVERIFIED unless you run this on your catalogue with
local CLIP weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ai.clip_model_paths import clip_model_available_locally
from src.ai.search_quality.description_colour_fusion import (
    extract_colour_tokens,
    fuse_clip_with_colour_rrf,
)
from src.ai.similarity_score import calibrate_description_search_display_map
from src.ai.tile_color_detect import detect_color_from_image
from src.core.use_cases.clip_text_query import (
    apply_relevance_cutoff,
    average_l2_normalized,
    description_prompt_variants,
)

COLOUR_QUERIES = ("gray", "white", "black", "beige", "brown")
FREE_TEXT_QUERIES = (
    "matte subway tile",
    "wood look plank",
    "marble veining",
    "hexagon mosaic",
    "glossy kitchen backsplash",
)
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.tif", "*.tiff")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_images(catalogue: Path) -> List[Path]:
    paths: List[Path] = []
    for pattern in IMAGE_EXTS:
        paths.extend(catalogue.rglob(pattern))
        paths.extend(catalogue.rglob(pattern.upper()))
    return sorted({p.resolve() for p in paths if p.is_file()})


def _dedupe_by_sha(paths: Sequence[Path]) -> List[Tuple[Path, str]]:
    seen: Dict[str, Path] = {}
    for p in paths:
        try:
            digest = _sha256(p)
        except OSError:
            continue
        if digest not in seen:
            seen[digest] = p
    return [(seen[digest], digest) for digest in sorted(seen)]


def precision_at_k(ranked_labels: Sequence[str], wanted: set[str], k: int) -> float:
    top = ranked_labels[:k]
    if not top:
        return 0.0
    hits = sum(1 for lab in top if lab in wanted)
    return hits / float(k)


def _load_clip():
    if not clip_model_available_locally():
        print(
            "CLIP weights are not available under model_weights/.\n"
            "Download openai/clip-vit-base-patch32 into model_weights before running eval.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    from src.ai.clip_embedder import ClipEmbedder

    clip = ClipEmbedder()
    clip.ensure_loaded()
    return clip


def _embed_images(clip, items: Sequence[Tuple[Path, str]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    total = len(items)
    for i, (path, sha) in enumerate(items, start=1):
        if sha in out:
            continue
        vec = clip.embed_image(str(path))
        out[sha] = np.asarray(vec, dtype=np.float32).reshape(-1)
        if i % 5 == 0 or i == total:
            print(f"  embedded {i}/{total} unique images…", flush=True)
    return out


def _rank_ids(
    query_vec: np.ndarray,
    id_list: List[int],
    matrix: np.ndarray,
) -> List[int]:
    scores = matrix @ query_vec
    order = np.argsort(-scores)
    return [id_list[int(i)] for i in order]


def _labels_for_ids(ids: List[int], colors: Dict[int, str]) -> List[str]:
    return [colors[i].lower() for i in ids]


def main() -> int:
    parser = argparse.ArgumentParser(description="Description-search catalogue eval")
    parser.add_argument(
        "--catalogue",
        type=Path,
        required=True,
        help="Root folder of tile images (Windows paths OK)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path",
    )
    args = parser.parse_args()
    catalogue = args.catalogue
    if not catalogue.is_dir():
        print(f"Catalogue not found: {catalogue}", file=sys.stderr)
        return 1

    print(f"Scanning {catalogue}…")
    paths = _collect_images(catalogue)
    if not paths:
        print("No images found.", file=sys.stderr)
        return 1
    unique = _dedupe_by_sha(paths)
    print(f"Found {len(paths)} files, {len(unique)} unique by sha256.")

    clip = _load_clip()
    print("Embedding unique images (once each)…")
    embeddings = _embed_images(clip, unique)

    ids: List[int] = []
    colors: Dict[int, str] = {}
    sha_by_id: Dict[int, str] = {}
    matrix_rows: List[np.ndarray] = []
    for idx, (path, sha) in enumerate(unique):
        tid = idx + 1
        ids.append(tid)
        sha_by_id[tid] = sha
        colors[tid] = detect_color_from_image(path)
        v = embeddings[sha]
        v = v / max(np.linalg.norm(v), 1e-12)
        matrix_rows.append(v)
    matrix = np.stack(matrix_rows, axis=0)

    queries = list(COLOUR_QUERIES) + list(FREE_TEXT_QUERIES)
    rows_out: List[dict] = []

    def run_mode(
        name: str,
        query: str,
        ranked: List[int],
        raw_scores: List[float],
    ) -> None:
        wanted = {query.lower()}
        if query.lower() in ("gray", "grey"):
            wanted = {"gray", "grey"}
        p5 = precision_at_k(_labels_for_ids(ranked, colors), wanted, 5)
        p10 = precision_at_k(_labels_for_ids(ranked, colors), wanted, 10)
        disp = calibrate_description_search_display_map(raw_scores[: len(ranked)])
        row = {
            "query": query,
            "mode": name,
            "p_at_5": round(p5, 4),
            "p_at_10": round(p10, 4),
            "display_top": disp[0] if disp else "",
            "display_min": min(disp) if disp else "",
            "score_top": round(raw_scores[0], 4) if raw_scores else "",
            "score_median": round(float(np.median(raw_scores)), 4) if raw_scores else "",
        }
        rows_out.append(row)
        print(
            f"  {name:28} P@5={p5:.3f} P@10={p10:.3f} "
            f"display_top={row['display_top']} score_top={row['score_top']}"
        )

    for q in queries:
        print(f"\nQuery: {q}")
        raw_vec = clip.embed_text(q)
        raw_vec = raw_vec / max(np.linalg.norm(raw_vec), 1e-12)
        ranked_raw = _rank_ids(raw_vec, ids, matrix)
        scores_raw = [float((matrix[i - 1] @ raw_vec)) for i in ranked_raw]
        run_mode("raw", q, ranked_raw, scores_raw)

        tmpl_vecs = [clip.embed_text(t) for t in description_prompt_variants(q)]
        ens = average_l2_normalized(tmpl_vecs)
        ranked_tmpl = _rank_ids(ens, ids, matrix)
        scores_tmpl = [float((matrix[i - 1] @ ens)) for i in ranked_tmpl]
        run_mode("templates", q, ranked_tmpl, scores_tmpl)

        pairs = [(i, scores_tmpl[pos]) for pos, i in enumerate(ranked_tmpl)]
        kept, _fb, _before = apply_relevance_cutoff(pairs, top_k=len(ids))
        order_cut = [tid for tid, _ in kept]
        scores_cut = [sc for _, sc in kept]
        run_mode("templates+cutoff", q, order_cut, scores_cut)

        tokens = extract_colour_tokens(q)
        if tokens:
            fused = fuse_clip_with_colour_rrf(pairs, colors, tokens)
            kept_f, _fb2, _ = apply_relevance_cutoff(fused, top_k=len(ids))
            order_f = [tid for tid, _ in kept_f]
            scores_f = [sc for _, sc in kept_f]
            run_mode("templates+cutoff+colour", q, order_f, scores_f)
        else:
            run_mode("templates+cutoff+colour", q, order_cut, scores_cut)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"\nWrote {args.csv}")

    print(
        "\nNOTE: Precision numbers above are from THIS catalogue run only. "
        "Mark UNVERIFIED when comparing across machines without repeating the run."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
