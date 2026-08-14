"""
Debug-only helpers to visualize index-time crop selection.

Does not import torch / DINOv2 — safe to run without model weights.

Uses ``prepare_index_primary`` — the same function production indexing calls —
so saved primary letterboxes match what ``extract_index_vectors`` embeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.ai.descriptors.color_descriptor import ColorDescriptor
from src.ai.descriptors.edge_descriptor import EdgeDescriptor
from src.ai.descriptors.pattern_descriptor import PatternDescriptor
from src.ai.descriptors.texture_descriptor import TextureDescriptor
from src.ai.models import TileFeatures
from src.ai.preprocess.image_preprocessor import ImagePreprocessor
from src.ai.preprocess.index_primary import (
    IndexPrimaryPreparation,
    prepare_index_primary,
)
from src.ai.search_quality.image_analysis import ImageAnalysis

QueryMode = Literal["auto", "catalog", "fresh", "both"]


@dataclass(frozen=True, slots=True)
class DescriptorParity:
    """One descriptor-similarity measurement between two feature sets."""

    mode: str
    color: float
    texture: float
    edge: float
    pattern: float
    query_letterbox_path: str
    index_letterbox_path: str
    notes: str


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
    parity_alt: DescriptorParity | None = None


def show_index_crops(
    image_path: str | Path,
    *,
    output_dir: str | Path = "/tmp/index_crop_debug",
    feature_extractor: Any | None = None,
    query_path: str | Path | None = None,
    query_mode: QueryMode = "auto",
    catalog_repo: Any | None = None,
) -> IndexCropReport:
    """
    Re-run the production index-time primary prep and save crop PNGs.

    The primary letterbox is taken from ``prepare_index_primary()`` — identical
    to ``FeatureExtractor.extract_index_vectors``. Debug-only: does not mutate
    the FAISS index.

    ``query_mode`` controls which hybrid comparison is printed:

    - ``catalog`` — simulate UI catalog-query cache hit: compare
      ``prepare_index_primary(query)`` descriptors vs index primary
      (index-time vs index-time). This is what production uses when the
      log says ``Reusing indexed features for catalog query``.
    - ``fresh`` — ad-hoc upload path: ``preprocess_for_query(query)`` vs
      index primary (can diverge when scene-isolation fires on non-square
      clean tiles).
    - ``auto`` / ``both`` — print catalog as primary; also print fresh.
      If ``catalog_repo`` is set and the query path is an indexed tile,
      catalog mode uses **stored SQLite TileFeatures** (byte-identical to
      SearchTilesUseCase) instead of a live recompute.
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

    primary_path = out / f"{stem}_primary_preprocess_letterbox.png"
    prep.primary.pil.save(primary_path)
    saved.append(str(primary_path))

    if prep.primary_source == "panel":
        legacy = ImagePreprocessor.preprocess(prep.source_path)
        legacy_path = out / f"{stem}_legacy_fullsheet_letterbox.png"
        legacy.pil.save(legacy_path)
        saved.append(str(legacy_path))

    parity: DescriptorParity | None = None
    parity_alt: DescriptorParity | None = None
    if query_path is not None:
        parity, parity_alt, extra_paths = _descriptor_parity_bundle(
            query_path=query_path,
            index_primary=prep,
            output_dir=out,
            catalog_stem=stem,
            query_mode=query_mode,
            catalog_repo=catalog_repo,
            catalog_image_path=prep.source_path,
        )
        saved.extend(extra_paths)

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
        parity_alt=parity_alt,
    )


def _sims_from_features(query: TileFeatures, candidate: TileFeatures) -> tuple[float, float, float, float]:
    return (
        float(
            ColorDescriptor.similarity(
                query.color_histogram, candidate.color_histogram
            )
        ),
        float(
            TextureDescriptor.similarity(
                query.texture_histogram, candidate.texture_histogram
            )
        ),
        float(
            EdgeDescriptor.similarity(
                query.edge_histogram, candidate.edge_histogram
            )
        ),
        float(
            PatternDescriptor.similarity(
                query.pattern_features, candidate.pattern_features
            )
        ),
    )


def _features_from_preprocessed(pre) -> TileFeatures:
    return TileFeatures(
        embedding=np.zeros(8, dtype=np.float32),
        color_histogram=ColorDescriptor.extract(pre.bgr),
        texture_histogram=TextureDescriptor.extract(pre.bgr),
        edge_histogram=EdgeDescriptor.extract(pre.bgr),
        pattern_features=PatternDescriptor.extract(pre.bgr),
        dominant_color=ColorDescriptor.dominant_color_rgb(pre.bgr),
        width=pre.width,
        height=pre.height,
    )


def _lookup_stored_features(repo: Any, image_path: Path) -> TileFeatures | None:
    try:
        from src.utils.image_utils import compute_sha256

        tile = repo.get_by_path(str(image_path.resolve()))
        if (
            tile is not None
            and tile.is_indexed
            and tile.features is not None
            and tile.sha256_hash == compute_sha256(image_path)
        ):
            return tile.features
    except Exception:
        return None
    return None


def _descriptor_parity_bundle(
    *,
    query_path: str | Path,
    index_primary: IndexPrimaryPreparation,
    output_dir: Path,
    catalog_stem: str,
    query_mode: QueryMode,
    catalog_repo: Any | None,
    catalog_image_path: str,
) -> tuple[DescriptorParity | None, DescriptorParity | None, list[str]]:
    qpath = Path(query_path).expanduser().resolve()
    if not qpath.is_file():
        raise FileNotFoundError(f"Query image not found: {qpath}")

    index_letter_path = (
        output_dir / f"{catalog_stem}_primary_preprocess_letterbox.png"
    )
    extra: list[str] = []

    # --- catalog / stored-vs-stored path ---
    query_index_prep = prepare_index_primary(qpath)
    q_index_letter = (
        output_dir / f"{qpath.stem}_index_primary_preprocess_letterbox.png"
    )
    query_index_prep.primary.pil.save(q_index_letter)
    extra.append(str(q_index_letter))

    stored_query = (
        _lookup_stored_features(catalog_repo, qpath) if catalog_repo else None
    )
    stored_cand = (
        _lookup_stored_features(catalog_repo, Path(catalog_image_path))
        if catalog_repo
        else None
    )

    if stored_query is not None and stored_cand is not None:
        c_color, c_tex, c_edge, c_pat = _sims_from_features(
            stored_query, stored_cand
        )
        catalog_notes = (
            "STORED SQLite TileFeatures (query) vs STORED (candidate) — "
            "exact SearchTilesUseCase catalog-cache hybrid components"
        )
        catalog_mode_label = "catalog_stored"
    else:
        q_feats = _features_from_preprocessed(query_index_prep.primary)
        i_feats = _features_from_preprocessed(index_primary.primary)
        c_color, c_tex, c_edge, c_pat = _sims_from_features(q_feats, i_feats)
        catalog_notes = (
            "prepare_index_primary(query) vs prepare_index_primary(candidate) — "
            "simulates catalog-cache when DB features unavailable. "
            "Pass --catalog to compare actual stored blobs."
        )
        catalog_mode_label = "catalog_sim"

    catalog_parity = DescriptorParity(
        mode=catalog_mode_label,
        color=c_color,
        texture=c_tex,
        edge=c_edge,
        pattern=c_pat,
        query_letterbox_path=str(q_index_letter),
        index_letterbox_path=str(index_letter_path),
        notes=catalog_notes,
    )

    # --- fresh ad-hoc query path ---
    query_pre = ImagePreprocessor.preprocess_for_query(qpath)
    q_fresh_letter = output_dir / f"{qpath.stem}_query_preprocess_letterbox.png"
    query_pre.pil.save(q_fresh_letter)
    extra.append(str(q_fresh_letter))

    fresh_q = _features_from_preprocessed(query_pre)
    fresh_i = _features_from_preprocessed(index_primary.primary)
    f_color, f_tex, f_edge, f_pat = _sims_from_features(fresh_q, fresh_i)
    fresh_parity = DescriptorParity(
        mode="fresh",
        color=f_color,
        texture=f_tex,
        edge=f_edge,
        pattern=f_pat,
        query_letterbox_path=str(q_fresh_letter),
        index_letterbox_path=str(index_letter_path),
        notes=(
            "preprocess_for_query(query) vs live index primary — ad-hoc upload "
            "path only. Non-square clean tiles may run scene isolation here, "
            "which catalog-cache skips. Do NOT treat as UI production when "
            "the log says 'Reusing indexed features for catalog query'."
        ),
    )

    mode = query_mode
    if mode == "auto":
        mode = "both"

    if mode == "catalog":
        return catalog_parity, None, extra
    if mode == "fresh":
        return fresh_parity, None, extra
    # both
    return catalog_parity, fresh_parity, extra


def format_index_crop_report(report: IndexCropReport) -> str:
    a = report.analysis
    lines = [
        f"Index crop debug: {report.source_path}",
        f"Output dir: {report.output_dir}",
        "",
        "IMPORTANT — production parity notes:",
        "  • Primary letterbox is from prepare_index_primary() — the SAME",
        "    function FeatureExtractor.extract_index_vectors() uses.",
        "  • SAM2 / Precise Crop is NOT part of catalog indexing.",
        "  • When the UI searches an already-indexed file (xx.jpg.jpeg), it",
        "    REUSES stored index-time TileFeatures — it does NOT run",
        "    preprocess_for_query. Log line:",
        "      'Reusing indexed features for catalog query: …'",
        "    That stored-vs-stored comparison is mode=catalog_* below.",
        "  • mode=fresh (preprocess_for_query) is the ad-hoc upload path and",
        "    can read much higher edge/pattern on real marble (scene",
        "    isolation / different crop). That is NOT the UI catalog path.",
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

    def _emit(p: DescriptorParity, title: str) -> None:
        lines.extend(
            [
                "",
                f"{title} [mode={p.mode}]:",
                f"  color={p.color:.3f}  texture={p.texture:.3f}  "
                f"edge={p.edge:.3f}  pattern={p.pattern:.3f}",
                f"  {p.notes}",
                f"  query_letterbox: {p.query_letterbox_path}",
                f"  index_letterbox:  {p.index_letterbox_path}",
            ]
        )

    if report.parity is not None:
        _emit(report.parity, "Descriptor parity (PRIMARY — matches UI when catalog)")
    if report.parity_alt is not None:
        _emit(report.parity_alt, "Descriptor parity (ALTERNATE — ad-hoc fresh path)")
    if report.parity is None:
        lines.extend(
            [
                "",
                "Descriptor parity: NOT computed (pass --query PATH).",
            ]
        )

    lines.append("")
    lines.append("Saved PNGs:")
    for path in report.saved_paths:
        lines.append(f"  {path}")
    return "\n".join(lines)
