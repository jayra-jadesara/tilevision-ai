"""
Image preprocessing utilities for TileVision AI.

Centralizes image loading and preprocessing so DINOv2 and handcrafted
descriptors operate on the same representation.

Pipeline
--------
load -> EXIF transpose -> alpha composite -> optional border trim
     -> aspect-ratio-preserving letterbox resize -> OpenCV arrays
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.ai.models import PreprocessedImage

logger = logging.getLogger("tilevision.ai.image_preprocessor")

# DINOv2 ViT patch size is 14; 518 = 37 * 14.
TARGET_SIZE = 518

# Default max edge when decoding huge catalogue masters (70–200 MB files).
DEFAULT_MAX_DECODE_EDGE = 2048

# Neutral pad color — avoids biasing color descriptors toward white/black.
PAD_COLOR: Tuple[int, int, int] = (128, 128, 128)


@dataclass(slots=True)
class PreprocessConfig:
    """Runtime preprocessing limits (wired from AppSettings at startup)."""

    max_decode_edge: int = DEFAULT_MAX_DECODE_EDGE


class ImagePreprocessor:
    """Shared preprocessing pipeline for all AI feature extractors."""

    DEFAULT_SIZE: Tuple[int, int] = (TARGET_SIZE, TARGET_SIZE)
    _config = PreprocessConfig()

    @classmethod
    def configure(cls, *, max_decode_edge: int | None = None) -> None:
        """Update module-level decode limits (called once from app startup)."""
        if max_decode_edge is not None:
            cls._config.max_decode_edge = max(512, int(max_decode_edge))

    @classmethod
    def max_decode_edge(cls) -> int:
        return cls._config.max_decode_edge

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        max_decode_edge: int | None = None,
    ) -> Image.Image:
        """
        Load an image, applying early downscale for huge catalogue masters.

        Opens the file once. Decodes to at most ``max_decode_edge`` on the
        longest side before border/crop/AI work — critical for 70–200 MB
        tile photography.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        max_edge = max_decode_edge or cls._config.max_decode_edge

        try:
            with Image.open(path) as img:
                fmt = img.format
                # Prefer draft() for JPEG/WebP so huge masters never fully
                # decode at native resolution (still a single open).
                if (
                    fmt in ("JPEG", "MPO", "WEBP")
                    and hasattr(img, "draft")
                    and max(img.size) > max_edge
                ):
                    img.draft("RGB", (max_edge, max_edge))

                image = ImageOps.exif_transpose(img)
                width, height = image.size

                if max(width, height) > max_edge:
                    image = image.copy()
                    image.thumbnail(
                        (max_edge, max_edge),
                        Image.Resampling.BICUBIC,
                    )
                    logger.debug(
                        "Early downscale %s: %dx%d -> %dx%d (max_edge=%d)",
                        path.name,
                        width,
                        height,
                        image.size[0],
                        image.size[1],
                        max_edge,
                    )
                    return image

                return image.copy()
        except Exception as e:
            logger.exception("Failed to load image: %s", path)
            raise RuntimeError(str(e)) from e

    @staticmethod
    def to_rgb(image: Image.Image) -> Image.Image:
        """
        Convert to RGB, compositing alpha onto a neutral background.
        """
        if image.mode in ("RGBA", "LA"):
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, PAD_COLOR)
            background.paste(rgba, mask=rgba.split()[-1])
            return background

        if image.mode == "P" and "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, PAD_COLOR)
            background.paste(rgba, mask=rgba.split()[-1])
            return background

        if image.mode != "RGB":
            return image.convert("RGB")

        return image

    @staticmethod
    def _is_uniform_border_row(row: np.ndarray, tolerance: int = 18) -> bool:
        """True when a row is a near-uniform light border (catalog background)."""
        if row.size == 0:
            return False
        mean = row.mean(axis=0)
        spread = np.max(np.abs(row.astype(np.int16) - mean.astype(np.int16)), axis=1)
        return bool(np.mean(spread <= tolerance) > 0.92 and mean.mean() > 200)

    @classmethod
    def trim_uniform_borders(
        cls,
        image: Image.Image,
        max_trim_ratio: float = 0.12,
    ) -> Image.Image:
        """
        Conservatively crop uniform white/light catalogue borders.

        Does nothing when no clear border is detected.
        """
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        if height < 32 or width < 32:
            return image

        max_v_trim = int(height * max_trim_ratio)
        max_h_trim = int(width * max_trim_ratio)

        top = 0
        for i in range(max_v_trim):
            if not cls._is_uniform_border_row(rgb[i]):
                break
            top = i + 1

        bottom = height
        for i in range(max_v_trim):
            if not cls._is_uniform_border_row(rgb[height - 1 - i]):
                break
            bottom = height - 1 - i

        left = 0
        for i in range(max_h_trim):
            if not cls._is_uniform_border_row(rgb[:, i]):
                break
            left = i + 1

        right = width
        for i in range(max_h_trim):
            if not cls._is_uniform_border_row(rgb[:, width - 1 - i]):
                break
            right = width - 1 - i

        if right - left < width * 0.5 or bottom - top < height * 0.5:
            return image

        if top == 0 and bottom == height and left == 0 and right == width:
            return image

        logger.debug(
            "Trimmed uniform borders: top=%d bottom=%d left=%d right=%d",
            top,
            bottom,
            left,
            right,
        )
        return image.crop((left, top, right, bottom))

    @classmethod
    def crop_to_content_region(
        cls,
        image: Image.Image,
        min_margin_ratio: float = 0.08,
    ) -> Image.Image:
        """
        Crop to the dominant textured region when clear background margins exist.

        Uses edge/variance detection — conservative: returns the original image
        when no confident content bounding box is found.
        """
        rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        if height < 48 or width < 48:
            return image

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        edges = cv2.Canny(blur, 40, 120)

        blur_f = blur.astype(np.float32)
        sq_blur = cv2.GaussianBlur(blur_f * blur_f, (15, 15), 0)
        variance = np.maximum(sq_blur - blur_f * blur_f, 0.0)
        texture_mask = (variance > 25.0).astype(np.uint8) * 255

        mask = cv2.bitwise_or(edges, texture_mask)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return image

        x, y, box_w, box_h = cv2.boundingRect(coords)
        if box_w < width * 0.35 or box_h < height * 0.35:
            return image

        margin_x = min(x, width - (x + box_w))
        margin_y = min(y, height - (y + box_h))
        if margin_x < width * min_margin_ratio and margin_y < height * min_margin_ratio:
            return image

        logger.debug(
            "Content-region crop: x=%d y=%d w=%d h=%d",
            x,
            y,
            box_w,
            box_h,
        )
        return image.crop((x, y, x + box_w, y + box_h))

    @classmethod
    def normalize_lighting(cls, image: Image.Image) -> Image.Image:
        """
        Mild LAB L-channel stretch for underexposed / crushed photos.

        A narrow L-channel range is *not* enough to decide to stretch:
        well-lit cream/white marble also has a compressed L-range (that is
        the material). Stretching those frames posterizes subtle veins
        (seen on PGYS2319 panel primary after v14 routed isolated panels
        through this path).

        Stretch only when the frame looks underexposed or crushed — dark
        mean and/or highlights well below white — not when it is already
        high-key with low chroma.
        """
        rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        low, high = np.percentile(l_channel, (2, 98))
        span = float(high - low)
        if span >= 40.0:
            return image

        mean_l = float(l_channel.mean())
        a_f = a_channel.astype(np.float32)
        b_f = b_channel.astype(np.float32)
        chroma = float(np.mean(np.hypot(a_f - 128.0, b_f - 128.0)))

        # High-key, low-chroma material (cream marble, white ceramic): leave
        # alone — narrow span is intrinsic, not a lighting defect.
        if mean_l >= 160.0 and high >= 195.0 and chroma <= 28.0:
            logger.debug(
                "normalize_lighting: skip high-key low-chroma material "
                "(mean_L=%.1f high=%.1f span=%.1f chroma=%.1f)",
                mean_l,
                high,
                span,
                chroma,
            )
            return image

        # Adequately bright frame even with some chroma — do not invent
        # contrast on already well-exposed product photography.
        if mean_l >= 170.0 and high >= 200.0:
            logger.debug(
                "normalize_lighting: skip bright well-exposed frame "
                "(mean_L=%.1f high=%.1f span=%.1f)",
                mean_l,
                high,
                span,
            )
            return image

        stretched = np.clip(
            (l_channel.astype(np.float32) - low) * (255.0 / max(span, 1.0)),
            0,
            255,
        ).astype(np.uint8)
        logger.info(
            "normalize_lighting: stretch underexposed/crushed frame "
            "(mean_L=%.1f high=%.1f span=%.1f chroma=%.1f)",
            mean_l,
            high,
            span,
            chroma,
        )
        merged = cv2.merge([stretched, a_channel, b_channel])
        corrected_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        corrected_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(corrected_rgb)

    @classmethod
    def _looks_like_scene_photo(cls, image: Image.Image) -> bool:
        """Heuristic: non-square framing or strong border/center difference."""
        width, height = image.size
        if width < 48 or height < 48:
            return False

        aspect = width / max(height, 1)
        if aspect < 0.88 or aspect > 1.14:
            return True

        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        margin_x = max(1, int(width * 0.12))
        margin_y = max(1, int(height * 0.12))
        if margin_x * 2 >= width or margin_y * 2 >= height:
            return False

        center = rgb[margin_y : height - margin_y, margin_x : width - margin_x]
        if center.size == 0:
            return False

        border_strips = [
            rgb[:margin_y, :],
            rgb[height - margin_y :, :],
            rgb[:, :margin_x],
            rgb[:, width - margin_x :],
        ]
        border = np.concatenate(
            [strip.reshape(-1, 3) for strip in border_strips],
            axis=0,
        )
        center_mean = center.reshape(-1, 3).mean(axis=0)
        border_mean = border.mean(axis=0)
        return float(np.linalg.norm(center_mean - border_mean)) >= 22.0

    @classmethod
    def primary_texture_panel(cls, image: Image.Image) -> Image.Image | None:
        """
        For wide catalog/marketing sheets, return the dominant tile panel.

        Showroom sheets often put a large slab on the left and text/grid on
        the right. Indexing only the full sheet makes texture-only customer
        crops (e.g. 600×600 from the slab) miss the parent image in FAISS.
        Returning the left panel lets us store a second FAISS vector under
        the same tile id. Ordinary square/portrait tiles return None.
        """
        image = image.convert("RGB")
        width, height = image.size
        if width < 480 or height < 320:
            return None
        rgb = np.asarray(image, dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        from src.ai.search_quality.image_analysis import marketing_sheet_panel_eligible

        if not marketing_sheet_panel_eligible(width, height, gray):
            return None

        # Take the left ~45% (not a full half) so the text/grid column does not
        # bleed into the panel. Then trim + content-crop for slab-only pixels.
        split_x = max(1, int(width * 0.45))
        left = image.crop((0, 0, split_x, height))
        left = cls.trim_uniform_borders(left)
        left = cls.crop_to_content_region(left, min_margin_ratio=0.02)
        pw, ph = left.size
        if pw < 64 or ph < 64:
            logger.debug(
                "primary_texture_panel: rejected small panel %sx%s",
                pw,
                ph,
            )
            return None

        # Qingyu-style sheets (PGYS2319): caption lines bleed across the top
        # panel edge into the top-left corner. panel_center (72% inset) clears
        # them; shave the caption band from the full panel aux crop boundary.
        # Real PGYS2319: 10% (v12) left a partial clipped line in top ~15px;
        # 13% clears it at 2x zoom without eating bottom marble texture.
        _PANEL_TOP_CAPTION_BAND_RATIO = 0.13
        top_cut = max(0, int(ph * _PANEL_TOP_CAPTION_BAND_RATIO))
        left_cut = max(0, int(pw * 0.03))
        if top_cut > 0 or left_cut > 0:
            new_w = pw - left_cut
            new_h = ph - top_cut
            if new_w >= 64 and new_h >= 64:
                left = left.crop((left_cut, top_cut, pw, ph))
                pw, ph = left.size
            else:
                top_cut = left_cut = 0

        arr = np.asarray(left, dtype=np.float32)
        # High-key white marble/onyx panels often have std ≈ 1–4. The old
        # threshold of 6.0 rejected every Qingyu-style slab (customer PGYS2319).
        panel_std = float(arr.std())
        if panel_std < 0.85:
            logger.info(
                "primary_texture_panel: rejected near-blank panel (std=%.3f)",
                panel_std,
            )
            return None
        # Require the panel to differ from a resized full sheet (avoid noop).
        full = np.asarray(image.resize(left.size), dtype=np.float32)
        mean_abs = float(np.mean(np.abs(arr - full)))
        if mean_abs < 3.0:
            logger.info(
                "primary_texture_panel: rejected panel too similar to full "
                "sheet (mean_abs=%.3f)",
                mean_abs,
            )
            return None
        logger.info(
            "primary_texture_panel: using left panel %sx%s (std=%.2f, Δfull=%.2f"
            "%s)",
            pw,
            ph,
            panel_std,
            mean_abs,
            f", top_cut={top_cut} left_cut={left_cut}"
            if top_cut or left_cut
            else "",
        )
        return left

    @classmethod
    def focus_center_region(
        cls,
        image: Image.Image,
        ratio: float = 0.72,
    ) -> Image.Image:
        """Crop to the central region — common tile location in room photos."""
        width, height = image.size
        crop_w = max(1, int(width * ratio))
        crop_h = max(1, int(height * ratio))
        left = (width - crop_w) // 2
        top = (height - crop_h) // 2
        return image.crop((left, top, left + crop_w, top + crop_h))

    @classmethod
    def resize_letterbox(
        cls,
        image: Image.Image,
        target: int = TARGET_SIZE,
        *,
        pad_color: Tuple[int, int, int] | None = None,
    ) -> Image.Image:
        """
        Resize preserving aspect ratio, then center-pad to a square canvas.

        ``pad_color`` defaults to ``PAD_COLOR`` (neutral gray). For tall
        catalog-panel crops, pass the panel mean color so gray pads do not
        pollute LAB color / texture histograms (~45% of a portrait panel
        letterbox can otherwise be pad pixels).
        """
        width, height = image.size
        if width <= 0 or height <= 0:
            return image

        scale = min(target / width, target / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        resized = image.resize(
            (new_width, new_height),
            Image.Resampling.BICUBIC,
        )

        fill = pad_color if pad_color is not None else PAD_COLOR
        canvas = Image.new("RGB", (target, target), fill)
        offset_x = (target - new_width) // 2
        offset_y = (target - new_height) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas

    @classmethod
    def resize(
        cls,
        image: Image.Image,
        size: Tuple[int, int] | None = None,
    ) -> Image.Image:
        """Backward-compatible alias for letterbox resize."""
        target = size[0] if size else TARGET_SIZE
        return cls.resize_letterbox(image, target=target)

    @staticmethod
    def to_numpy(image: Image.Image) -> np.ndarray:
        return np.asarray(image, dtype=np.uint8)

    @staticmethod
    def to_bgr(image: Image.Image) -> np.ndarray:
        rgb = np.asarray(image, dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @classmethod
    def preprocess(
        cls,
        image_path: str | Path,
    ) -> PreprocessedImage:
        image = cls.load(image_path)
        original_width, original_height = image.size

        image = cls.to_rgb(image)
        image = cls.trim_uniform_borders(image)
        image = cls.crop_to_content_region(image)
        image = cls.normalize_lighting(image)
        image = cls.resize_letterbox(image)

        rgb = cls.to_numpy(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        return PreprocessedImage(
            pil=image,
            rgb=rgb,
            bgr=bgr,
            gray=gray,
            width=original_width,
            height=original_height,
        )

    @classmethod
    def preprocess_for_query(
        cls,
        image_path: str | Path,
        *,
        preloaded: Image.Image | None = None,
    ) -> PreprocessedImage:
        """
        Search-only preprocessing with extra handling for scene/room photos.

        Does not change the indexing pipeline or feature_version — only
        applied at query time to improve room-photo searches.

        Room photos use fast OpenCV tile-region isolation (no SAM on drop),
        then optional perspective straighten. Clean catalogue tiles skip that
        path. Use Precise Crop & Search for ONNX SAM2 on hard room photos.

        Pass ``preloaded`` to avoid a second disk decode when the caller
        already opened the image (dHash / validation).
        """
        path = Path(image_path)
        image = preloaded if preloaded is not None else cls.load(path)
        original_width, original_height = image.size

        image = cls.to_rgb(image)
        image = cls.trim_uniform_borders(image)
        image = cls.crop_to_content_region(image, min_margin_ratio=0.05)

        already_cropped = "tilevision_crops" in path.as_posix().lower()
        is_catalog_sheet = False
        if not already_cropped and cls._looks_like_scene_photo(image):
            # v1.2.32: catalogue sheets need white-margin/text evidence.
            # ``primary_texture_panel is not None`` false-triggered on room
            # photos (wide aspect + textured left third) and skipped isolation
            # — measured cosine ~0.47–0.53 vs parent; isolation recovers ~0.86+.
            from src.ai.search_quality.query_analyzer import QueryKind, analyze_query

            qanalysis = analyze_query(image)
            is_catalog_sheet = qanalysis.kind == QueryKind.CATALOG_SHEET
            if is_catalog_sheet:
                logger.info(
                    "Query catalog marketing sheet detected — skipping scene "
                    "auto-crop and perspective straighten"
                )
            else:
                if qanalysis.kind == QueryKind.PHONE_SCREENSHOT:
                    from src.ai.search_quality.query_views import strip_phone_ui

                    image = strip_phone_ui(image)
                image = cls._isolate_query_tile(image)

        # Crop-tool temp files are already isolated. Perspective warp on a
        # frontal marble crop invents a quad from vein lines and distorts
        # the embedding (measured 356×326 → 328×328 on floor_band output).
        if not is_catalog_sheet and not already_cropped:
            image = cls._maybe_straighten(image)
        image = cls.normalize_lighting(image)
        image = cls.resize_letterbox(image)

        rgb = cls.to_numpy(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        return PreprocessedImage(
            pil=image,
            rgb=rgb,
            bgr=bgr,
            gray=gray,
            width=original_width,
            height=original_height,
        )

    @classmethod
    def prepare_query_views(
        cls,
        image_path: str | Path,
        *,
        max_views: int = 3,
        preloaded: Image.Image | None = None,
    ) -> list[PreprocessedImage]:
        """
        Build 1–N query views for multi-crop embedding (search-only).

        Primary view matches ``preprocess_for_query``. Extra views come from
        alternate OpenCV tile candidates when the photo looks like a room scene.
        On Mac (Intel + Silicon) and Windows CPU, cap extras so DINOv2 query
        latency stays reasonable and search does not feel stuck.

        Pass ``preloaded`` so drop-search can decode the query file once.
        """
        path = Path(image_path)
        try:
            max_views = cls._capped_query_max_views(int(max_views))
        except Exception:
            max_views = min(int(max_views), 2)

        primary = cls.preprocess_for_query(path, preloaded=preloaded)
        views = [primary]

        # Drop-search uses max_views=1 — never re-decode for unused extra crops.
        if max_views <= 1:
            return views

        if "tilevision_crops" in path.as_posix().lower():
            return views

        # Extra views need an unletterboxed RGB working copy. Reuse the
        # caller's preloaded image when available; otherwise load once.
        image = preloaded.copy() if preloaded is not None else cls.load(path)
        image = cls.to_rgb(image)
        image = cls.trim_uniform_borders(image)
        image = cls.crop_to_content_region(image, min_margin_ratio=0.05)
        if not cls._looks_like_scene_photo(image):
            return views

        from src.ai.preprocess.fast_tile_crop import list_tile_region_candidates

        candidates = list_tile_region_candidates(image, limit=max(1, int(max_views)))
        original_width, original_height = image.size
        for crop in candidates[1:max_views]:
            view = cls._finalize_query_pil(
                crop.image,
                original_width=original_width,
                original_height=original_height,
            )
            views.append(view)

        logger.info("Query multi-crop views prepared: %d", len(views))
        return views

    @classmethod
    def _capped_query_max_views(cls, requested: int) -> int:
        """
        Limit multi-crop DINOv2 work when query inference is effectively CPU.

        - Mac Intel: always CPU
        - Mac Silicon: query path forces CPU to avoid MPS hangs
        - Windows without CUDA: CPU showroom PCs
        CUDA Windows keeps the full requested view count.
        """
        requested = max(1, int(requested))
        from src.utils.platform_info import is_macos, is_windows

        if is_macos():
            return min(requested, 2)
        if is_windows():
            try:
                from src.ai.gpu_info import detect_gpu_runtime

                if detect_gpu_runtime(preference="auto").active_device == "cuda":
                    return requested
            except Exception:
                pass
            return min(requested, 2)
        return min(requested, 2)

    @classmethod
    def _isolate_query_tile(cls, image: Image.Image) -> Image.Image:
        """
        Fast OpenCV isolation for default drop-search (all platforms).

        SAM2 Precise Crop stays on the explicit Precise Crop & Search button
        so Windows / Mac Intel / Mac Silicon drop-search stays responsive.
        """
        from src.ai.preprocess.fast_tile_crop import isolate_tile_region

        crop = isolate_tile_region(image)
        logger.info(
            "Query scene auto-crop: method=%s confidence=%.2f size=%dx%d",
            crop.method,
            crop.confidence,
            crop.image.size[0],
            crop.image.size[1],
        )
        return crop.image

    @classmethod
    def _maybe_straighten(cls, image: Image.Image) -> Image.Image:
        try:
            from src.ai.preprocess.perspective_straighten import straighten_tile_view

            return straighten_tile_view(image)
        except Exception as exc:
            logger.debug("Perspective straighten skipped: %s", exc)
            return image

    @classmethod
    def _finalize_query_pil(
        cls,
        image: Image.Image,
        *,
        original_width: int,
        original_height: int,
        straighten: bool = True,
    ) -> PreprocessedImage:
        image = image.convert("RGB")
        if straighten:
            image = cls._maybe_straighten(image)
        image = cls.normalize_lighting(image)
        image = cls.resize_letterbox(image)
        rgb = cls.to_numpy(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return PreprocessedImage(
            pil=image,
            rgb=rgb,
            bgr=bgr,
            gray=gray,
            width=original_width,
            height=original_height,
        )
