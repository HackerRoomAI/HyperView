"""Public API for HyperView."""

from __future__ import annotations

import webbrowser

import uvicorn

from hyperview.core.dataset import Dataset
from hyperview.server.app import create_app, set_dataset

__all__ = ["Dataset", "launch"]


def launch(
    dataset: Dataset,
    port: int = 5151,
    host: str = "127.0.0.1",
    open_browser: bool = True,
) -> None:
    """Launch the HyperView visualization server.

    Args:
        dataset: The dataset to visualize.
        port: Port to run the server on.
        host: Host to bind to.
        open_browser: Whether to open a browser window.

    Example:
        >>> import hyperview as hv
        >>> dataset = hv.Dataset("my_dataset")
        >>> dataset.add_images_dir("/path/to/images", label_from_folder=True)
        >>> dataset.compute_embeddings()
        >>> dataset.compute_visualization()
        >>> hv.launch(dataset)
    """
    set_dataset(dataset)
    app = create_app(dataset)

    url = f"http://{host}:{port}"
    print(f"\n🚀 HyperView is running at {url}")
    print("   Press Ctrl+C to stop.\n")

    if open_browser:
        webbrowser.open(url)

    uvicorn.run(app, host=host, port=port, log_level="warning")
