"""Embedding computation using EmbedAnything."""

from __future__ import annotations

import os
import tempfile

import embed_anything
import numpy as np
from embed_anything import EmbeddingModel, WhichModel
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from hyperview.core.sample import Sample


class EmbeddingComputer:
    """Compute embeddings for images using EmbedAnything."""

    def __init__(self, model: str = "clip"):
        """Initialize the embedding computer.

        Args:
            model: Model to use for embeddings ('clip' supported).
        """
        self.model_name = model
        self._model = None
        self._initialized = False

    def _init_model(self) -> None:
        """Lazily initialize the model."""
        if self._initialized:
            return

        # Use CLIP model for image embeddings
        self._model = EmbeddingModel.from_pretrained_hf(
            WhichModel.Clip,
            model_id="openai/clip-vit-base-patch32",
        )
        self._embed_anything = embed_anything
        self._initialized = True

    def _load_rgb_image(self, sample: Sample) -> Image.Image:
        """Load an image and ensure it is in RGB mode."""
        image = sample.load_image()
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _embed_with_model(
        self,
        sample: Sample,
        image: Image.Image | None = None,
    ) -> np.ndarray | None:
        """Attempt to embed a sample via embed_anything, handling memory-backed files."""
        path = sample.filepath
        temp_path: str | None = None

        try:
            if path.startswith("memory://"):
                if image is None:
                    image = self._load_rgb_image(sample)
                temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                image.save(temp_file, format="PNG")
                temp_file.close()
                temp_path = temp_file.name
                path = temp_path

            result = self._embed_anything.embed_file(path, embedder=self._model)
            if result:
                return np.array(result[0].embedding, dtype=np.float32)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        return None

    def compute_single(self, sample: Sample) -> np.ndarray:
        """Compute embedding for a single sample.

        Args:
            sample: Sample to compute embedding for.

        Returns:
            Embedding as numpy array.
        """
        self._init_model()

        pil_image = None
        if sample.filepath.startswith("memory://"):
            pil_image = self._load_rgb_image(sample)

        embedding = self._embed_with_model(sample, image=pil_image)
        if embedding is not None:
            return embedding

        if pil_image is None:
            pil_image = self._load_rgb_image(sample)

        # Fallback: compute from PIL image using numpy-based approach
        return self._compute_from_pil(pil_image)

    def _compute_from_pil(self, image) -> np.ndarray:
        """Compute a simple embedding from PIL image.

        This is a fallback when embed_anything can't process the image directly.
        Uses average color and basic features as a placeholder.
        """
        # Resize to consistent size
        img = image.resize((224, 224), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0

        # Compute simple features
        features = []

        # Average color per channel
        features.extend(arr.mean(axis=(0, 1)).flatten())

        # Std per channel
        features.extend(arr.std(axis=(0, 1)).flatten())

        # Grid features (4x4 grid averages)
        grid_size = 4
        h, w = arr.shape[:2]
        gh, gw = h // grid_size, w // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                cell = arr[i * gh : (i + 1) * gh, j * gw : (j + 1) * gw]
                features.extend(cell.mean(axis=(0, 1)).flatten())

        # Pad or truncate to 512 dimensions
        features = np.array(features)
        if len(features) < 512:
            features = np.pad(features, (0, 512 - len(features)))
        else:
            features = features[:512]

        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features

    def compute_batch(
        self,
        samples: list[Sample],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> list[np.ndarray]:
        """Compute embeddings for a batch of samples.

        Args:
            samples: List of samples to compute embeddings for.
            batch_size: Number of samples to process at once.
            show_progress: Whether to show a progress bar.

        Returns:
            List of embeddings as numpy arrays.
        """
        self._init_model()

        embeddings = []
        total = len(samples)

        if show_progress and tqdm is not None:
            iterator = tqdm(range(0, total, batch_size), desc="Computing embeddings")
        else:
            if show_progress and tqdm is None:
                print(f"Computing embeddings for {total} samples...")
            iterator = range(0, total, batch_size)

        for i in iterator:
            batch = samples[i : i + batch_size]
            batch_embeddings = []

            for sample in batch:
                pil_image = None
                if sample.filepath.startswith("memory://"):
                    pil_image = self._load_rgb_image(sample)

                embedding = self._embed_with_model(sample, image=pil_image)
                if embedding is None:
                    if pil_image is None:
                        pil_image = self._load_rgb_image(sample)
                    embedding = self._compute_from_pil(pil_image)

                batch_embeddings.append(embedding)

            embeddings.extend(batch_embeddings)

        return embeddings
