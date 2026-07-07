"""
OpenCLIP embedder module for TileVision AI.

Handles loading OpenCLIP models and performing offline feature extraction (inference)
to generate high-dimensional image embeddings.
"""

import logging
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import torch

try:
    import open_clip
except ImportError:
    # Fallback/mock support for environments where open_clip is not pre-installed yet
    open_clip = None

logger = logging.getLogger("tilevision.ai.embedder")


class OpenCLIPEmbedder:
    """
    Service wrapper around OpenCLIP models using PyTorch.
    
    Generates L2-normalized 512-dimensional (or model-specific) float32 embeddings.
    """

    def __init__(self, model_name: str, pretrained: str) -> None:
        """
        Initialize the embedder with model config.

        Args:
            model_name: Name of the CLIP model (e.g. "ViT-B-32-quickgelu").
            pretrained: Path to local weights file or name of pretrained dataset.
        """
        self._model_name = model_name
        self._pretrained = pretrained
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._preprocess = None
        
        logger.info(f"OpenCLIPEmbedder initialized. Hardware device: {self._device.upper()}")

    def load_model(self) -> None:
        """
        Load the CLIP model and preprocessing transforms.
        
        Supports loading local weight files for offline installations.
        """
        if open_clip is None:
            logger.critical("open_clip package is not installed! Cannot load model.")
            raise ImportError("open-clip-torch package is required for OpenCLIPEmbedder.")

        logger.info(f"Loading CLIP model '{self._model_name}' (weights source: {self._pretrained})...")
        try:
            # If pretrained is a local file path, pass it directly
            weights_path = Path(self._pretrained)
            if weights_path.exists() and weights_path.is_file():
                logger.info(f"Loading local model weights from file: {weights_path}")
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self._model_name,
                    pretrained=str(weights_path.resolve()),
                    device=self._device,
                )
            else:
                # Load via open_clip download manager (will download if internet is active,
                # or load from ~/.cache/clip offline if already cached)
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self._model_name,
                    pretrained=self._pretrained,
                    device=self._device,
                )
            
            self._model = model
            self._preprocess = preprocess
            
            # Set model to evaluation mode
            self._model.eval()
            
            logger.info("CLIP model successfully loaded.")
        except Exception as e:
            logger.critical(f"Failed to load CLIP model weights: {e}")
            raise RuntimeError(f"Model load error: {e}") from e

    def get_embedding(self, image_path: str) -> List[float]:
        """
        Generate L2-normalized feature vector for the given image.

        Args:
            image_path: Path to the image file.

        Returns:
            A list of floats representing the normalized feature vector.
        """
        if self._model is None or self._preprocess is None:
            self.load_model()

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        try:
            # Load and preprocess image
            with Image.open(path) as img:
                # Convert RGBA/grayscale to RGB
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Apply transforms and add batch dimension
                image_tensor = self._preprocess(img).unsqueeze(0).to(self._device)

            # Perform inference without gradient tracking
            with torch.no_grad():
                # Extract image features
                image_features = self._model.encode_image(image_tensor)
                
                # L2 normalize the embedding
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Move to CPU and convert to numpy list
                embedding_np = image_features.cpu().numpy().flatten().astype(np.float32)
                
            return list(embedding_np.tolist())
        except Exception as e:
            logger.error(f"Failed to extract embedding for {image_path}: {e}")
            raise RuntimeError(f"Embedding extraction error: {e}") from e
