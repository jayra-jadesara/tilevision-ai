"""
Debug-only helpers to visualize index-time crop selection.

Does not import torch / DINOv2 — safe to run without model weights.

Uses ``prepare_index_primary`` — the same function production indexing calls —
so saved primary letterboxes match what ``extract_index_vectors`` embeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.ai.descriptors.color_descriptor import ColorDescriptor
from src.ai.descriptors.edge_descriptor import EdgeDescriptor
from src.ai.descriptors.pattern_descriptor import PatternDescriptor
from src.ai.descriptors.texture_descriptor import TextureDescriptor
from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.preprocess.index_primary import (
    IndexPrimaryPreparation,
    prepare_index_primary,
)
from src.ai.search_quality.image_analysis import ImageAnalysis


@dataclass(frozen=True, slots=True)
class DescriptorParity:
    """Handcrafted descriptor similarities: query preprocess vs index primary."""

    color: float
    texture: float
    edge: float
    pattern: float
    query_letterbox_path: str
    index_letterbox_path: str


@dataclass(frozen=True, slots=True)
class IndexCropReport:
    source_path: str
    output_dir: str
    analysis: ImageAnalysis
    saved_paths: tuple[str, ...]
    index_view_types: tuple[str, ...]
    primary_panel: dict[str, Any] | None
    primary_source: str
    parity: DescriptorParity | None = None


def show_index_crops(
    image_path: str | Path,
    *,
    output_dir: str | Path = "/tmp/index_crop_debug",
    feature_extractor: Any | None = None,
    query_path: str | Path | None = None,
) -> IndexCropReport:
    """
    Re-run the production index-time primary prep and save crop PNGs.

    The primary letterbox is taken from ``prepare_index_primary()`` — identical
    to ``FeatureExtractor.extract_index_vectors``. Debug-only: does not mutate
    the FAISS index.

    Pass ``query_path`` to also save the query-time letterbox
    (``preprocess_for_query``) and print descriptor similarities that match
    hybrid component scores (except embedding / pattern-compat / penalties).
    """
    prep = prepare_index_primary(image_path)
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    stem = Path(prep.source_path).stem
    saved: list[str] = []
    view_types: list[str] = []

    for idx, view in enumerate(prep.views):
        label = f"{stem}_view{idx}_{view.view_type.value}"
        dest = out / f"{label}.png"
        view.image.save(dest)
        saved.append(str(dest))
        view_types.append(view.view_type.value)

    panel_info: dict[str, Any] | None = None
    if prep.panel is not None:
        arr = np.asarray(prep.panel, dtype=np.float32)
        full = np.asarray(prep.raw.resize(prep.panel.size), dtype=np.float32)
        panel_info = {
            "size": prep.panel.size,
            "std": float(arr.std()),
            "mean_abs_delta_full": float(np.mean(np.abs(arr - full))),
        }
        panel_path = out / f"{stem}_primary_texture_panel.png"
        prep.panel.save(panel_path)
        saved.append(str(panel_path))

    # Exact production letterbox bytes (not a parallel reconstruction).
    primary_path = out / f"{stem}_primary_preprocess_letterbox.png"
    prep.primary.pil.save(primary_path)
    saved.append(str(primary_path))

    if prep.primary_source == "panel":
        legacy = ImagePreprocessor.preprocess(prep.source_path)
        legacy_path = out / f"{stem}_legacy_fullsheet_letterbox.png"
        legacy.pil.save(legacy_path)
        saved.append(str(legacy_path))

    parity: DescriptorParity | None = None
    if query_path is not None:
        parity = _descriptor_parity_vs_query(
            query_path=query_path,
            index_primary=prep,
            output_dir=out,
            catalog_stem=stem,
        )
        saved.append(parity.query_letterbox_path)

    if feature_extractor is not None:
        _, aux = feature_extractor.extract_index_vectors(str(prep.source_path))
        aux_path = out / f"{stem}_aux_vectors.txt"
        aux_path.write_text(
            f"aux_vector_count={len(aux)}\n"
            f"primary_source={prep.primary_source}\n"
            + "\n".join(
                f"aux_{i}_dim={len(v)} norm={float(np.linalg.norm(v)):.4f}"
                for i, v in enumerate(aux)
            )
            + "\n",
            encoding="utf-8",
        )
        saved.append(str(aux_path))

    return IndexCropReport(
        source_path=prep.source_path,
        output_dir=str(out),
        analysis=prep.analysis,
        saved_paths=tuple(saved),
        index_view_types=tuple(view_types),
        primary_panel=panel_info,
        primary_source=prep.primary_source,
        parity=parity,
    )


def _descriptor_parity_vs_query(
    *,
    query_path: str | Path,
    index_primary: IndexPrimaryPreparation,
    output_dir: Path,
    catalog_stem: str,
) -> DescriptorParity:
    qpath = Path(query_path).expanduser().resolve()
    if not qpath.is_file():
        raise FileNotFoundError(f"Query image not found: {qpath}")

    query_pre = ImagePreprocessor.preprocess_for_query(qpath)
    q_letter_path = output_dir / f"{qpath.stem}_query_preprocess_letterbox.png"
    query_pre.pil.save(q_letter_path)

    index_letter_path = (
        output_dir / f"{catalog_stem}_primary_preprocess_letterbox.png"
    )
    q_bgr = query_pre.bgr
    i_bgr = index_primary.primary.bgr

    return DescriptorParity(
        color=float(
            ColorDescriptor.similarity(
                ColorDescriptor.extract(q_bgr),
                ColorDescriptor.extract(i_bgr),
            )
        ),
        texture=float(
            TextureDescriptor.similarity(
                TextureDescriptor.extract(q_bgr),
                TextureDescriptor.extract(i_bgr),
            )
        ),
        edge=float(
            EdgeDescriptor.similarity(
                EdgeDescriptor.extract(q_bgr),
                EdgeDescriptor.extract(i_bgr),
            )
        ),
        pattern=float(
            PatternDescriptor.similarity(
                PatternDescriptor.extract(q_bgr),
                PatternDescriptor.extract(i_bgr),
            )
        ),
        query_letterbox_path=str(q_letter_path),
        index_letterbox_path=str(index_letter_path),
    )


def format_index_crop_report(report: IndexCropReport) -> str:
    a = report.analysis
    lines = [
        f"Index crop debug: {report.source_path}",
        f"Output dir: {report.output_dir}",
        "",
        "IMPORTANT — production parity notes:",
        "  • Primary letterbox is from prepare_index_primary() — the SAME",
        "    function FeatureExtractor.extract_index_vectors() uses.",
        "  • SAM2 / Precise Crop is NOT part of catalog indexing. The",
        "    startup log 'SAM2 precise-crop setting ON' only enables the",
        "    UI 'Precise Crop & Search' button (and optional scene path).",
        "  • Hybrid component scores in explain_search compare",
        "    preprocess_for_query(query) vs indexed primary descriptors —",
        "    NOT two index letterboxes. Pass --query to measure that pair.",
        "  • Re-reading saved PNGs and re-extracting can differ slightly",
        "    from in-memory arrays (PNG round-trip); prefer --query output.",
        "",
        "Image analysis (index-time gates):",
        f"  kind={a.kind.value}  aspect={a.aspect:.3f}  "
        f"texture_richness={a.texture_richness:.3f}",
        f"  text_region_score={a.text_region_score:.3f}  "
        f"has_preview_grid={a.has_preview_grid}",
        f"  white_border_ratio={a.white_border_ratio:.3f}",
        f"  left_panel_beneficial={a.left_panel_beneficial}  "
        f"center_crop_beneficial={a.center_crop_beneficial}",
        f"  quality_score={a.quality_score:.3f}",
        "",
        f"Index views ({len(report.index_view_types)}): "
        + ", ".join(report.index_view_types),
        f"primary_source={report.primary_source}",
    ]
    if report.primary_panel:
        p = report.primary_panel
        lines.append(
            f"primary_texture_panel: {p['size'][0]}x{p['size'][1]} "
            f"std={p['std']:.3f} mean_abs_delta_full={p['mean_abs_delta_full']:.3f}"
        )
    else:
        lines.append("primary_texture_panel: None (no left-panel aux vector path)")
    if report.primary_source == "panel":
        lines.append(
            "primary_preprocess_letterbox: ISOLATED PANEL "
            "(feeds TileFeatures embedding + descriptors)"
        )
    else:
        lines.append(
            "primary_preprocess_letterbox: full sheet "
            "(feeds TileFeatures embedding + descriptors)"
        )

    if report.parity is not None:
        p = report.parity
        lines.extend(
            [
                "",
                "Descriptor parity (query preprocess ↔ index primary):",
                f"  color={p.color:.3f}  texture={p.texture:.3f}  "
                f"edge={p.edge:.3f}  pattern={p.pattern:.3f}",
                "  These should match explain_search hybrid components for the",
                "  same pair (embedding / compat / color_penalty are separate).",
                f"  query_letterbox: {p.query_letterbox_path}",
                f"  index_letterbox:  {p.index_letterbox_path}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Descriptor parity: NOT computed (pass --query PATH to compare",
                "  against production hybrid color/texture/edge/pattern).",
            ]
        )

    lines.append("")
    lines.append("Saved PNGs:")
    for path in report.saved_paths:
        lines.append(f"  {path}")
    return "\n".join(lines)
