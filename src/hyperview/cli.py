"""Command-line interface for HyperView."""

from __future__ import annotations

import argparse
import sys

from hyperview import Dataset, launch


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="hyperview",
        description="HyperView - Dataset visualization with hyperbolic embeddings",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a demo with sample data")
    demo_parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of samples to load (default: 500)",
    )
    demo_parser.add_argument(
        "--port",
        type=int,
        default=5151,
        help="Port to run the server on (default: 5151)",
    )

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Serve a saved dataset")
    serve_parser.add_argument("dataset", help="Path to saved dataset JSON file")
    serve_parser.add_argument(
        "--port",
        type=int,
        default=5151,
        help="Port to run the server on (default: 5151)",
    )

    args = parser.parse_args()

    if args.command == "demo":
        run_demo(args.samples, args.port)
    elif args.command == "serve":
        serve_dataset(args.dataset, args.port)
    else:
        parser.print_help()
        sys.exit(1)


def run_demo(num_samples: int = 500, port: int = 5151):
    """Run a demo with CIFAR-100 data."""
    print("🔄 Loading CIFAR-100 dataset...")
    dataset = Dataset("cifar100_demo")

    try:
        count = dataset.add_from_huggingface(
            "uoft-cs/cifar100",
            split="train",
            image_key="img",
            label_key="fine_label",
            max_samples=num_samples,
        )
        print(f"✓ Loaded {count} samples")
    except Exception as e:
        print(f"Failed to load HuggingFace dataset: {e}")
        print("Please ensure 'datasets' is installed: pip install datasets")
        sys.exit(1)

    print("🔄 Computing embeddings...")
    dataset.compute_embeddings(show_progress=True)
    print("✓ Embeddings computed")

    print("🔄 Computing visualizations...")
    dataset.compute_visualization()
    print("✓ Visualizations ready")

    launch(dataset, port=port)


def serve_dataset(filepath: str, port: int = 5151):
    """Serve a saved dataset."""
    from hyperview import Dataset, launch

    print(f"🔄 Loading dataset from {filepath}...")
    dataset = Dataset.load(filepath)
    print(f"✓ Loaded {len(dataset)} samples")

    launch(dataset, port=port)


if __name__ == "__main__":
    main()
