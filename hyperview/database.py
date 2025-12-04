import lancedb
import numpy as np
import pandas as pd
import os

DB_PATH = "data/lancedb"

# Weather/scene labels like BDD100K
LABELS = ["clear", "foggy", "overcast", "partly cloudy", "rainy", "snowy"]

def generate_clustered_embeddings(num_points, num_clusters):
    """Generate embeddings with realistic cluster structure"""
    points_per_cluster = num_points // num_clusters
    euclidean_points = []
    hyperbolic_points = []
    labels = []

    for i in range(num_clusters):
        # Euclidean: clusters with gaussian distribution
        center_x = np.cos(2 * np.pi * i / num_clusters) * 2
        center_y = np.sin(2 * np.pi * i / num_clusters) * 2

        cluster_x = np.random.normal(center_x, 0.4, points_per_cluster)
        cluster_y = np.random.normal(center_y, 0.4, points_per_cluster)
        euclidean_points.extend(zip(cluster_x, cluster_y))

        # Hyperbolic: clusters in Poincare disk (preserve hierarchy)
        # Each cluster occupies an angular sector, with hierarchy from center to edge
        angle_center = 2 * np.pi * i / num_clusters
        angle_spread = np.random.normal(0, 0.3, points_per_cluster)
        angles = angle_center + angle_spread

        # Radii: mix of central (common) and peripheral (rare) samples
        # Use beta distribution to create realistic embedding structure
        radii = np.sqrt(np.random.beta(2, 5, points_per_cluster)) * 0.95

        hyp_x = radii * np.cos(angles)
        hyp_y = radii * np.sin(angles)
        hyperbolic_points.extend(zip(hyp_x, hyp_y))

        labels.extend([i] * points_per_cluster)

    return np.array(euclidean_points), np.array(hyperbolic_points), np.array(labels)

def init_db():
    os.makedirs(DB_PATH, exist_ok=True)
    db = lancedb.connect(DB_PATH)

    # Check if table exists
    if "images" not in db.table_names():
        print("Generating clustered mock data (like BDD100K)...")
        num_points = 1200  # 200 per label
        num_clusters = len(LABELS)

        euclidean, hyperbolic, labels = generate_clustered_embeddings(num_points, num_clusters)

        ids = [str(i) for i in range(num_points)]
        # Use picsum with different seeds for variety
        image_urls = [f"https://picsum.photos/seed/{i + labels[i] * 1000}/200/200" for i in range(num_points)]

        data = pd.DataFrame({
            "id": ids,
            "vector": list(euclidean.astype(np.float32)),
            "hyperbolic_x": hyperbolic[:, 0].astype(np.float32),
            "hyperbolic_y": hyperbolic[:, 1].astype(np.float32),
            "euclidean_x": euclidean[:, 0].astype(np.float32),
            "euclidean_y": euclidean[:, 1].astype(np.float32),
            "image_url": image_urls,
            "label": labels,
            "label_name": [LABELS[l] for l in labels]
        })

        db.create_table("images", data, mode="overwrite")
        print(f"Generated {num_points} points with {num_clusters} clusters.")

    return db

def get_table():
    db = lancedb.connect(DB_PATH)
    return db.open_table("images")
