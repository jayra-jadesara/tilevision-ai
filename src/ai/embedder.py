"""
DINOv2 embedder module for TileVision AI.

Uses Meta DINOv2 with multi-patch feature extraction.

The final embedding remains the same dimension as the original
DINOv2 model, allowing it to work directly with FAISS.

DINOv2 Large:
    1024 dimensions

Pipeline:

Image
  │
  ├── Full Image
  ├── Center Crop
  ├── Top Left
  ├── Top Right
  ├── Bottom Left
  └── Bottom Right
          │
          ▼
      DINOv2
          │
          ▼
    Average Embeddings
          │
          ▼
      L2 Normalize
          │
          ▼
       1024D Vector
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from PIL import Image

import torch
from transformers import AutoImageProcessor, AutoModel

from src.ai.preprocess.image_preprocessor import ImagePreprocessor


logger = logging.getLogger(
    "tilevision.ai.embedder"
)


class DINOv2Embedder:

    MODEL_NAME = "facebook/dinov2-large"

    def __init__(self) -> None:

        self._device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self._processor = None
        self._model = None

        logger.info(
            "DINOv2 Embedder initialized. Device: %s",
            self._device.type.upper(),
        )

    # ========================================================
    # Model loading
    # ========================================================

    def load_model(self) -> None:

        if self._model is not None:
            return

        logger.info(
            "Loading DINOv2 model..."
        )

        self._processor = (
            AutoImageProcessor.from_pretrained(
                self.MODEL_NAME
            )
        )

        self._model = (
            AutoModel.from_pretrained(
                self.MODEL_NAME
            )
        )

        self._model.to(
            self._device
        )

        self._model.eval()

        logger.info(
            "DINOv2 model loaded successfully."
        )

    # ========================================================
    # Single image embedding
    # ========================================================

    def _extract_single(
        self,
        image: Image.Image,
    ) -> np.ndarray:

        inputs = self._processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self._device)
            for key, value
            in inputs.items()
        }

        with torch.inference_mode():

            if self._device.type == "cuda":

                with torch.autocast(
                    device_type="cuda",
                ):

                    outputs = self._model(
                        **inputs
                    )

            else:

                outputs = self._model(
                    **inputs
                )

        # CLS token
        embedding = (
            outputs
            .last_hidden_state[:, 0]
            .cpu()
            .numpy()[0]
            .astype(np.float32)
        )

        # Normalize individual embedding
        embedding /= (
            np.linalg.norm(
                embedding
            )
            + 1e-8
        )

        return embedding

    # ========================================================
    # Generate patches
    # ========================================================

    @staticmethod
    def _generate_patches(
        image: Image.Image,
    ) -> List[Image.Image]:

        image = image.convert(
            "RGB"
        )

        width, height = image.size

        patches: List[Image.Image] = []

        # ----------------------------------------------------
        # 1. Full image
        # ----------------------------------------------------

        patches.append(
            image
        )

        # Don't create tiny patches.
        if width < 100 or height < 100:
            return patches

        # ----------------------------------------------------
        # Patch size
        #
        # Use approximately 65% of image dimensions.
        # This creates overlapping crops and retains context.
        # ----------------------------------------------------

        crop_width = max(
            1,
            int(width * 0.65),
        )

        crop_height = max(
            1,
            int(height * 0.65),
        )

        # ----------------------------------------------------
        # 2. Top-left
        # ----------------------------------------------------

        patches.append(
            image.crop(
                (
                    0,
                    0,
                    crop_width,
                    crop_height,
                )
            )
        )

        # ----------------------------------------------------
        # 3. Top-right
        # ----------------------------------------------------

        patches.append(
            image.crop(
                (
                    width - crop_width,
                    0,
                    width,
                    crop_height,
                )
            )
        )

        # ----------------------------------------------------
        # 4. Bottom-left
        # ----------------------------------------------------

        patches.append(
            image.crop(
                (
                    0,
                    height - crop_height,
                    crop_width,
                    height,
                )
            )
        )

        # ----------------------------------------------------
        # 5. Bottom-right
        # ----------------------------------------------------

        patches.append(
            image.crop(
                (
                    width - crop_width,
                    height - crop_height,
                    width,
                    height,
                )
            )
        )

        # ----------------------------------------------------
        # 6. Center
        # ----------------------------------------------------

        left = (
            width - crop_width
        ) // 2

        top = (
            height - crop_height
        ) // 2

        patches.append(
            image.crop(
                (
                    left,
                    top,
                    left + crop_width,
                    top + crop_height,
                )
            )
        )

        return patches

    # ========================================================
    # Multi-patch embedding
    # ========================================================

    def extract(
        self,
        image_path: str,
    ) -> np.ndarray:

        if self._model is None:
            self.load_model()

        processed = (
            ImagePreprocessor.preprocess(
                image_path
            )
        )

        image = processed.pil

        patches = (
            self._generate_patches(
                image
            )
        )

        embeddings = []

        for patch in patches:

            embedding = (
                self._extract_single(
                    patch
                )
            )

            embeddings.append(
                embedding
            )

        # Stack:
        #
        # 6 x 1024
        #
        stacked = np.stack(
            embeddings,
            axis=0,
        )

        # Average all local + global representations
        final_embedding = np.mean(
            stacked,
            axis=0,
        ).astype(
            np.float32
        )

        # Normalize final embedding
        final_embedding /= (
            np.linalg.norm(
                final_embedding
            )
            + 1e-8
        )

        logger.debug(
            "Multi-patch DINOv2 embedding: "
            "patches=%d dimension=%d",
            len(patches),
            final_embedding.shape[0],
        )

        return final_embedding

    # ========================================================
    # Backward compatibility
    # ========================================================

    def get_embedding(
        self,
        image_path: str,
    ) -> np.ndarray:

        return self.extract(
            image_path
        )