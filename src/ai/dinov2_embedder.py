from transformers import AutoImageProcessor
from transformers import AutoModel

import torch
import numpy as np
from PIL import Image


class DINOv2Embedder:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-large"
        )

        self.model = AutoModel.from_pretrained(
            "facebook/dinov2-large"
        )

        self.model.to(self.device)
        self.model.eval()