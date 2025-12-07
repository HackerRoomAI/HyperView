#!/usr/bin/env python3
"""Run HyperView demo with CIFAR-100 dataset."""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Run HyperView demo")
    parser.add_argument(
        "--samples", type=int, default=500, help="Number of samples to load (default: 500)"
    )
    parser.add_argument(
        "--port", type=int, default=5151, help="Port to run server on (default: 5151)"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't open browser automatically"
    )
    args = parser.parse_args()

    import hyperview as hv

    print(f"Loading {args.samples} samples from CIFAR-100...")
    dataset = hv.Dataset("cifar100_demo")
    count = dataset.add_from_huggingface(
        "uoft-cs/cifar100",
        split="train",
        image_key="img",
        label_key="fine_label",
        max_samples=args.samples,
    )
    print(f"Loaded {count} samples")

    print("Computing embeddings...")
    dataset.compute_embeddings(show_progress=True)

    print("Computing visualization (UMAP + Poincare)...")
    dataset.compute_visualization()

    print(f"Starting server at http://127.0.0.1:{args.port}")

    hv.launch(dataset, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
