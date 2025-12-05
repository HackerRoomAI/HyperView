"""Dataset class for managing collections of samples."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from PIL import Image

from hyperview.core.sample import Sample, SampleFromArray


class Dataset:
    """A collection of samples with support for embeddings and visualization."""

    def __init__(self, name: str | None = None):
        """Initialize a new dataset.

        Args:
            name: Optional name for the dataset.
        """
        self.name = name or f"dataset_{uuid.uuid4().hex[:8]}"
        self._samples: dict[str, Sample] = {}
        self._embedding_computer = None
        self._projection_engine = None
        self._label_colors: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples.values())

    def __getitem__(self, sample_id: str) -> Sample:
        return self._samples[sample_id]

    def add_sample(self, sample: Sample) -> None:
        """Add a sample to the dataset."""
        self._samples[sample.id] = sample
        if sample.label and sample.label not in self._label_colors:
            self._assign_label_color(sample.label)

    def add_image(
        self,
        filepath: str,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        sample_id: str | None = None,
    ) -> Sample:
        """Add a single image to the dataset.

        Args:
            filepath: Path to the image file.
            label: Optional label for the image.
            metadata: Optional metadata dictionary.
            sample_id: Optional custom ID. If not provided, one will be generated.

        Returns:
            The created Sample.
        """
        if sample_id is None:
            sample_id = hashlib.md5(filepath.encode()).hexdigest()[:12]

        sample = Sample(
            id=sample_id,
            filepath=filepath,
            label=label,
            metadata=metadata or {},
        )
        self.add_sample(sample)
        return sample

    def add_images_dir(
        self,
        directory: str,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
        label_from_folder: bool = False,
        recursive: bool = True,
    ) -> int:
        """Add all images from a directory.

        Args:
            directory: Path to the directory containing images.
            extensions: Tuple of valid file extensions.
            label_from_folder: If True, use parent folder name as label.
            recursive: If True, search subdirectories.

        Returns:
            Number of images added.
        """
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        count = 0
        pattern = "**/*" if recursive else "*"

        for path in directory.glob(pattern):
            if path.is_file() and path.suffix.lower() in extensions:
                label = path.parent.name if label_from_folder else None
                self.add_image(str(path), label=label)
                count += 1

        return count

    def add_from_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        image_key: str = "img",
        label_key: str | None = "fine_label",
        label_names_key: str | None = None,
        max_samples: int | None = None,
    ) -> int:
        """Load samples from a HuggingFace dataset.

        Args:
            dataset_name: Name of the HuggingFace dataset.
            split: Dataset split to use.
            image_key: Key for the image column.
            label_key: Key for the label column (can be None).
            label_names_key: Key for label names in dataset info.
            max_samples: Maximum number of samples to load.

        Returns:
            Number of samples added.
        """
        ds = load_dataset(dataset_name, split=split)

        # Get label names if available
        label_names = None
        if label_key and label_names_key:
            if label_names_key in ds.features:
                 label_names = ds.features[label_names_key].names
        elif label_key:
            if hasattr(ds.features[label_key], "names"):
                label_names = ds.features[label_key].names

        count = 0
        total = len(ds) if max_samples is None else min(len(ds), max_samples)

        for i in range(total):
            item = ds[i]
            image = item[image_key]

            # Handle PIL Image or numpy array
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image

            # Get label
            label = None
            if label_key and label_key in item:
                label_idx = item[label_key]
                if label_names and isinstance(label_idx, int):
                    label = label_names[label_idx]
                else:
                    label = str(label_idx)

            sample = SampleFromArray.from_array(
                id=f"{dataset_name.replace('/', '_')}_{split}_{i}",
                image_array=image_array,
                label=label,
                metadata={"source": dataset_name, "split": split, "index": i},
            )
            self.add_sample(sample)
            count += 1

        return count

    def compute_embeddings(
        self,
        model: str = "clip",
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> None:
        """Compute embeddings for all samples.

        Args:
            model: Embedding model to use ('clip' supported).
            batch_size: Batch size for processing.
            show_progress: Whether to show progress bar.
        """
        from hyperview.embeddings.compute import EmbeddingComputer

        if self._embedding_computer is None:
            self._embedding_computer = EmbeddingComputer(model=model)

        samples = list(self._samples.values())
        embeddings = self._embedding_computer.compute_batch(
            samples, batch_size=batch_size, show_progress=show_progress
        )

        for sample, embedding in zip(samples, embeddings):
            sample.embedding = embedding.tolist()

    def compute_visualization(
        self,
        method: str = "umap",
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
    ) -> None:
        """Compute 2D projections for visualization.

        Args:
            method: Projection method ('umap' supported).
            n_neighbors: Number of neighbors for UMAP.
            min_dist: Minimum distance for UMAP.
            metric: Distance metric for UMAP.
        """
        from hyperview.embeddings.projection import ProjectionEngine

        if self._projection_engine is None:
            self._projection_engine = ProjectionEngine()

        samples = [s for s in self._samples.values() if s.embedding is not None]
        if not samples:
            raise ValueError("No embeddings computed. Call compute_embeddings() first.")

        embeddings = np.array([s.embedding for s in samples])

        # Compute Euclidean 2D projection
        coords_euclidean = self._projection_engine.project_umap(
            embeddings,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
        )

        # Compute Hyperbolic (Poincaré) 2D projection
        coords_hyperbolic = self._projection_engine.project_to_poincare(
            embeddings,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )

        for sample, coord_e, coord_h in zip(samples, coords_euclidean, coords_hyperbolic):
            sample.embedding_2d = coord_e.tolist()
            sample.embedding_2d_hyperbolic = coord_h.tolist()

    def _assign_label_color(self, label: str) -> None:
        """Assign a color to a label."""
        # Use a predefined color palette
        colors = [
            "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
            "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
            "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
            "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
        ]
        idx = len(self._label_colors) % len(colors)
        self._label_colors[label] = colors[idx]

    def get_label_colors(self) -> dict[str, str]:
        """Get the color mapping for labels."""
        return self._label_colors.copy()

    @property
    def samples(self) -> list[Sample]:
        """Get all samples as a list."""
        return list(self._samples.values())

    @property
    def labels(self) -> list[str]:
        """Get unique labels in the dataset."""
        return list(set(s.label for s in self._samples.values() if s.label))

    def filter(self, predicate: Callable[[Sample], bool]) -> list[Sample]:
        """Filter samples based on a predicate function."""
        return [s for s in self._samples.values() if predicate(s)]

    def to_dict(self) -> dict[str, Any]:
        """Convert dataset to dictionary for serialization."""
        return {
            "name": self.name,
            "num_samples": len(self),
            "labels": self.labels,
            "label_colors": self._label_colors,
        }

    def save(self, filepath: str, include_thumbnails: bool = True) -> None:
        """Save dataset to a JSON file.

        Args:
            filepath: Path to save the JSON file.
            include_thumbnails: Whether to include cached thumbnails.
        """
        # Cache thumbnails before saving if requested
        if include_thumbnails:
            for s in self._samples.values():
                s.cache_thumbnail()

        data = {
            "name": self.name,
            "label_colors": self._label_colors,
            "samples": [
                {
                    "id": s.id,
                    "filepath": s.filepath,
                    "label": s.label,
                    "metadata": s.metadata,
                    "embedding": s.embedding,
                    "embedding_2d": s.embedding_2d,
                    "embedding_2d_hyperbolic": s.embedding_2d_hyperbolic,
                    "thumbnail_base64": s.thumbnail_base64 if include_thumbnails else None,
                }
                for s in self._samples.values()
            ],
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> Dataset:
        """Load dataset from a JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        dataset = cls(name=data["name"])
        dataset._label_colors = data.get("label_colors", {})

        for s_data in data["samples"]:
            sample = Sample(
                id=s_data["id"],
                filepath=s_data["filepath"],
                label=s_data.get("label"),
                metadata=s_data.get("metadata", {}),
                embedding=s_data.get("embedding"),
                embedding_2d=s_data.get("embedding_2d"),
                embedding_2d_hyperbolic=s_data.get("embedding_2d_hyperbolic"),
                thumbnail_base64=s_data.get("thumbnail_base64"),
            )
            dataset.add_sample(sample)

        return dataset
