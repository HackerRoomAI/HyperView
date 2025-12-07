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

    # Supported embedding models for images
    # These models work well with image data (photos, natural images)
    SUPPORTED_MODELS = {
        "clip-vit-base-patch32": {
            "model_id": "openai/clip-vit-base-patch32",
            "which_model": WhichModel.Clip,
            "display_name": "CLIP ViT-B/32",
            "description": "Fast, general-purpose vision model (512-dim)",
            "data_types": ["image"],
        },
        "clip-vit-base-patch16": {
            "model_id": "openai/clip-vit-base-patch16",
            "which_model": WhichModel.Clip,
            "display_name": "CLIP ViT-B/16",
            "description": "Higher resolution, better quality (512-dim)",
            "data_types": ["image"],
        },
        "clip-vit-large-patch14": {
            "model_id": "openai/clip-vit-large-patch14",
            "which_model": WhichModel.Clip,
            "display_name": "CLIP ViT-L/14",
            "description": "Best quality for general images (768-dim)",
            "data_types": ["image"],
        },
        "jina-clip-v1": {
            "model_id": "jinaai/jina-clip-v1",
            "which_model": WhichModel.Clip,
            "display_name": "Jina CLIP v1",
            "description": "Multilingual, supports 89 languages (768-dim)",
            "data_types": ["image"],
        },
        "siglip-base-patch16": {
            "model_id": "google/siglip-base-patch16-224",
            "which_model": WhichModel.Clip,
            "display_name": "SigLIP Base",
            "description": "Improved zero-shot classification (768-dim)",
            "data_types": ["image"],
        },
    }

    @classmethod
    def get_models_for_data_type(cls, data_type: str) -> dict[str, dict]:
        """Get models that support a specific data type.

        Args:
            data_type: Type of data ('image', 'text', 'audio', etc.)

        Returns:
            Dictionary of model names to their configurations.
        """
        return {
            name: config
            for name, config in cls.SUPPORTED_MODELS.items()
            if data_type in config.get("data_types", [])
        }

    def __init__(self, model: str = "clip-vit-base-patch32"):
        """Initialize the embedding computer.

        Args:
            model: Model to use for embeddings (see SUPPORTED_MODELS).
        """
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model}. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )
        self.model_name = model
        self._model = None
        self._initialized = False

    def _init_model(self) -> None:
        """Lazily initialize the model."""
        if self._initialized:
            return

        model_config = self.SUPPORTED_MODELS[self.model_name]
        self._model = EmbeddingModel.from_pretrained_hf(
            model_config["which_model"],
            model_id=model_config["model_id"],
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
        if embedding is None:
            raise RuntimeError(f"Failed to compute embedding for sample {sample.id}")

        return embedding

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
                    raise RuntimeError(
                        f"Failed to compute embedding for sample {sample.id}"
                    )

                batch_embeddings.append(embedding)

            embeddings.extend(batch_embeddings)

        return embeddings

