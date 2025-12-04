"""
Data loader for CIFAR-100 dataset with embedding generation and hyperbolic projection.
Uses the hierarchical structure (20 superclasses, 100 fine classes) to demonstrate
the benefits of hyperbolic embeddings.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import lancedb

# CIFAR-100 superclass (coarse) labels
SUPERCLASS_NAMES = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
    'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
    'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
    'trees', 'vehicles_1', 'vehicles_2'
]

# Fine label to superclass mapping
FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13
]

# CIFAR-100 fine label names
FINE_LABEL_NAMES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# Colors for superclasses (20 distinct colors)
SUPERCLASS_COLORS = [
    [0, 119, 190],    # aquatic_mammals - ocean blue
    [0, 180, 230],    # fish - light blue
    [255, 105, 180],  # flowers - pink
    [139, 69, 19],    # food_containers - brown
    [255, 165, 0],    # fruit_and_vegetables - orange
    [128, 128, 128],  # household_electrical - gray
    [160, 82, 45],    # household_furniture - sienna
    [255, 215, 0],    # insects - gold
    [255, 69, 0],     # large_carnivores - red-orange
    [70, 130, 180],   # large_man-made - steel blue
    [34, 139, 34],    # large_natural - forest green
    [210, 105, 30],   # large_omnivores - chocolate
    [244, 164, 96],   # medium_mammals - sandy brown
    [138, 43, 226],   # non-insect_invertebrates - purple
    [255, 182, 193],  # people - light pink
    [50, 205, 50],    # reptiles - lime green
    [255, 140, 0],    # small_mammals - dark orange
    [0, 100, 0],      # trees - dark green
    [220, 20, 60],    # vehicles_1 - crimson
    [65, 105, 225],   # vehicles_2 - royal blue
]


def exponential_map_poincare(v: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Map Euclidean vectors to Poincaré ball using exponential map at origin.

    This is the key function that converts standard embeddings to hyperbolic space.
    Formula: exp_0(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
    """
    sqrt_c = np.sqrt(c)
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    v_norm = np.clip(v_norm, 1e-10, None)

    # Compute tanh factor for exponential map
    factor = np.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
    poincare_points = factor * v

    # Ensure points are strictly inside the ball (|x| < 1)
    norms = np.linalg.norm(poincare_points, axis=1, keepdims=True)
    poincare_points = np.where(norms >= 0.99, poincare_points * 0.99 / norms, poincare_points)

    return poincare_points


def project_to_poincare_disk(embeddings: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """
    Project high-dimensional embeddings to 2D Poincaré disk.
    Uses PCA for dimensionality reduction, then exponential map.
    """
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    embeddings_2d = embeddings_2d / np.std(embeddings_2d) * scale
    return exponential_map_poincare(embeddings_2d)


class FeatureExtractor:
    """Extract features using a pre-trained ResNet model."""

    def __init__(self, device: str = None):
        self.device = device or ('mps' if torch.backends.mps.is_available() else
                                 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load pre-trained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_batch(self, images: list) -> np.ndarray:
        """Extract features from a batch of PIL images."""
        batch = torch.stack([self.transform(img) for img in images]).to(self.device)
        features = self.model(batch)
        return features.squeeze(-1).squeeze(-1).cpu().numpy()


def create_cifar100_dataset(
    output_dir: str = './data/lancedb',
    data_root: str = './data',
    num_samples: int = 2000,
    force_regenerate: bool = False
):
    """
    Create a HyperView dataset from CIFAR-100.

    CIFAR-100 has a natural 2-level hierarchy:
    - 20 superclasses (coarse)
    - 100 fine classes

    This hierarchy is perfect for demonstrating hyperbolic embeddings.
    """
    os.makedirs(output_dir, exist_ok=True)
    db = lancedb.connect(output_dir)

    if "images" in db.table_names() and not force_regenerate:
        print("Dataset already exists. Use force_regenerate=True to recreate.")
        return

    print("=" * 60)
    print("Creating HyperView Dataset from CIFAR-100")
    print("=" * 60)

    # Download CIFAR-100
    print("\nDownloading CIFAR-100...")
    dataset = CIFAR100(root=data_root, train=True, download=True)

    # Sample stratified by superclass
    print(f"\nSelecting {num_samples} samples stratified by superclass...")
    np.random.seed(42)

    # Group indices by superclass
    superclass_indices = {i: [] for i in range(20)}
    for idx in range(len(dataset)):
        fine_label = dataset.targets[idx]
        coarse_label = FINE_TO_COARSE[fine_label]
        superclass_indices[coarse_label].append(idx)

    # Sample equally from each superclass
    samples_per_class = num_samples // 20
    selected_indices = []
    for coarse_id in range(20):
        indices = superclass_indices[coarse_id]
        selected = np.random.choice(indices, min(samples_per_class, len(indices)), replace=False)
        selected_indices.extend(selected)

    selected_indices = selected_indices[:num_samples]
    np.random.shuffle(selected_indices)

    # Extract images and labels
    print(f"Loading {len(selected_indices)} images...")
    images = []
    fine_labels = []
    coarse_labels = []

    for idx in selected_indices:
        img, fine_label = dataset[idx]
        images.append(img)
        fine_labels.append(fine_label)
        coarse_labels.append(FINE_TO_COARSE[fine_label])

    # Extract features
    print("\nExtracting features with ResNet18...")
    extractor = FeatureExtractor()
    batch_size = 64
    all_features = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        features = extractor.extract_batch(batch)
        all_features.append(features)
        print(f"  Processed {min(i+batch_size, len(images))}/{len(images)}")

    embeddings = np.vstack(all_features)
    print(f"Embedding shape: {embeddings.shape}")

    # Project to 2D Euclidean
    print("\nProjecting to 2D Euclidean space (PCA)...")
    pca = PCA(n_components=2)
    euclidean_2d = pca.fit_transform(embeddings)
    euclidean_2d = euclidean_2d / np.std(euclidean_2d) * 2

    # Project to Poincaré disk
    print("Projecting to Poincaré disk (hyperbolic)...")
    poincare_2d = project_to_poincare_disk(embeddings, scale=0.6)

    # Save images to disk
    print("\nSaving images...")
    image_dir = Path(data_root) / 'hyperview_images'
    image_dir.mkdir(exist_ok=True)

    image_urls = []
    for i, img in enumerate(images):
        # CIFAR images are 32x32, upscale to 128x128 for better viewing
        img_resized = img.resize((128, 128), Image.LANCZOS)
        img_path = image_dir / f'{i:05d}.jpg'
        img_resized.save(img_path, 'JPEG', quality=90)
        image_urls.append(f'/api/images/{i:05d}.jpg')

    # Create DataFrame
    print("\nCreating database...")
    df = pd.DataFrame({
        'id': [str(i) for i in range(len(images))],
        'vector': list(embeddings.astype(np.float32)),
        'euclidean_x': euclidean_2d[:, 0].astype(np.float32),
        'euclidean_y': euclidean_2d[:, 1].astype(np.float32),
        'hyperbolic_x': poincare_2d[:, 0].astype(np.float32),
        'hyperbolic_y': poincare_2d[:, 1].astype(np.float32),
        'image_url': image_urls,
        'label': coarse_labels,  # Use superclass as primary label
        'fine_label': fine_labels,
        'label_name': [SUPERCLASS_NAMES[c] for c in coarse_labels],
        'fine_label_name': [FINE_LABEL_NAMES[f] for f in fine_labels],
    })

    # Create table
    if "images" in db.table_names():
        db.drop_table("images")
    db.create_table("images", df)

    # Save label info
    label_info = pd.DataFrame({
        'label_id': list(range(20)),
        'label_name': SUPERCLASS_NAMES,
        'color_r': [c[0] for c in SUPERCLASS_COLORS],
        'color_g': [c[1] for c in SUPERCLASS_COLORS],
        'color_b': [c[2] for c in SUPERCLASS_COLORS],
    })

    if "labels" in db.table_names():
        db.drop_table("labels")
    db.create_table("labels", label_info)

    print(f"\n{'=' * 60}")
    print("Dataset created successfully!")
    print(f"  - {len(images)} images")
    print(f"  - 20 superclasses, 100 fine classes")
    print(f"  - Images saved to: {image_dir}")
    print(f"  - Database saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    create_cifar100_dataset(
        output_dir='./data/lancedb',
        data_root='./data',
        num_samples=2000,
        force_regenerate=True
    )
