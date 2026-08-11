"""
Debug-only helpers to visualize index-time crop selection.

Does not import torch / DINOv2 — safe to run without model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.search_quality.image_analysis import ImageAnalysis, analyze_image
from src.ai.search_quality.views import IndexStrategy, build_index_views


@dataclass(frozen=True, slots=True)
class IndexCropReport:
    source_path: str
    output_dir: str
    analysis: ImageAnalysis
    saved_paths: tuple[str, ...]
    index_view_types: tuple[str, ...]
    primary_panel: dict[str, Any] | None


def show_index_crops(
    image_path: str | Path,
    *,
    output_dir: str | Path = "/tmp/index_crop_debug",
    feature_extractor: Any | None = None,
) -> IndexCropReport:
    """
    Re-run the production index-time view plan and save embedded crop PNGs.

    Debug-only: does not mutate the FAISS index.
    """
    source = Path(image_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Catalog image not found: {source}")

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    raw = ImagePreprocessor.load(source)
    raw = ImagePreprocessor.to_rgb(raw)
    analysis = analyze_image(raw)
    views = build_index_views(
        raw,
        IndexStrategy.E_HEURISTIC_MULTIVIEW,
        analysis=analysis,
    )

    stem = source.stem
    saved: list[str] = []
    view_types: list[str] = []

    for idx, view in enumerate(views):
        label = f"{stem}_view{idx}_{view.view_type.value}"
        dest = out / f"{label}.png"
        view.image.save(dest)
        saved.append(str(dest))
        view_types.append(view.view_type.value)

    panel = ImagePreprocessor.primary_texture_panel(raw)
    panel_info: dict[str, Any] | None = None
    if panel is not None:
        arr = np.asarray(panel, dtype=np.float32)
        full = np.asarray(raw.resize(panel.size), dtype=np.float32)
        panel_info = {
            "size": panel.size,
            "std": float(arr.std()),
            "mean_abs_delta_full": float(np.mean(np.abs(arr - full))),
        }
        panel_path = out / f"{stem}_primary_texture_panel.png"
        panel.save(panel_path)
        saved.append(str(panel_path))

    # Primary letterbox must match extract_index_vectors: panel when beneficial.
    if analysis.left_panel_beneficial and panel is not None:
        primary_src = panel
        primary_note = "panel"
        mean = np.asarray(primary_src.convert("RGB"), dtype=np.float32).mean(
            axis=(0, 1)
        )
        pad_color = (
            int(np.clip(mean[0], 0, 255)),
            int(np.clip(mean[1], 0, 255)),
            int(np.clip(mean[2], 0, 255)),
        )
        primary_lit = ImagePreprocessor.normalize_lighting(primary_src)
        primary_letter = ImagePreprocessor.resize_letterbox(
            primary_lit,
            pad_color=pad_color,
        )
    else:
        primary_note = "full_sheet"
        primary_pre = ImagePreprocessor.preprocess(source)
        primary_letter = primary_pre.pil
    primary_path = out / f"{stem}_primary_preprocess_letterbox.png"
    primary_letter.save(primary_path)
    saved.append(str(primary_path))
    # Also save legacy full-sheet letterbox for before/after diffs.
    if primary_note == "panel":
        legacy = ImagePreprocessor.preprocess(source)
        legacy_path = out / f"{stem}_legacy_fullsheet_letterbox.png"
        legacy.pil.save(legacy_path)
        saved.append(str(legacy_path))

    if feature_extractor is not None:
        _, aux = feature_extractor.extract_index_vectors(str(source))
        aux_path = out / f"{stem}_aux_vectors.txt"
        aux_path.write_text(
            f"aux_vector_count={len(aux)}\n"
            f"primary_source={primary_note}\n"
            + "\n".join(
                f"aux_{i}_dim={len(v)} norm={float(np.linalg.norm(v)):.4f}"
                for i, v in enumerate(aux)
            )
            + "\n",
            encoding="utf-8",
        )
        saved.append(str(aux_path))

    return IndexCropReport(
        source_path=str(source),
        output_dir=str(out),
        analysis=analysis,
        saved_paths=tuple(saved),
        index_view_types=tuple(view_types),
        primary_panel=panel_info,
    )


def format_index_crop_report(report: IndexCropReport) -> str:
    a = report.analysis
    lines = [
        f"Index crop debug: {report.source_path}",
        f"Output dir: {report.output_dir}",
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
    ]
    if report.primary_panel:
        p = report.primary_panel
        lines.append(
            f"primary_texture_panel: {p['size'][0]}x{p['size'][1]} "
            f"std={p['std']:.3f} mean_abs_delta_full={p['mean_abs_delta_full']:.3f}"
        )
    else:
        lines.append("primary_texture_panel: None (no left-panel aux vector path)")
    if report.analysis.left_panel_beneficial and report.primary_panel:
        lines.append(
            "primary_preprocess_letterbox: ISOLATED PANEL "
            "(feeds TileFeatures embedding + descriptors)"
        )
    else:
        lines.append(
            "primary_preprocess_letterbox: full sheet "
            "(feeds TileFeatures embedding + descriptors)"
        )
    lines.append("")
    lines.append("Saved PNGs:")
    for path in report.saved_paths:
        lines.append(f"  {path}")
    return "\n".join(lines)
