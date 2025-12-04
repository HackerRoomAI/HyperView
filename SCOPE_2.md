- We need to move on from the hyperbolic embeddings demo and start turning this into a real product.
- It is a competitor to fiftyone
- Mainly we need to show the emebeddings next to the images
- See the following screenshot of the following link (https://try.fiftyone.ai/datasets/try-bdd/samples) once you add the embeddings tab
![fifty-one-embed](context/fiftyone_embeddings.png)
- the main features to replicate are: 
    - show images in one panel in a grid
    - show euclidean embedding in another panel as a scatter plot
    - allow selecting points in the scatter plot and see which images they correspond to
    - allow selecting images and see where they are in the scatter plot
- our twist is that we should allow an option to switch to hyperbolic embeddings

- We also need to get more serious about the architecture and using an actual dataset and actual embeddings
- For the architecture, we would need on python pacakge and one frontend
- the frontend can be a sepearate folder (e.g React/nextjs) that can be compiled to static files which can be exported and placed in the python package, export.sh needed, note it should be possible to develop the frontent without having to export it by running the server in the python package
- the python package would be pip installable and have a similar user flow to fiftyone, it should mainly use uv for the installation process
- for embeddings we would use embedanything package: https://github.com/StarlightSearch/EmbedAnything (make sure to check compativilty of this as it has some rust things)
- the rest of the frontend technologies need to be researched, we would like to use as much readily available things as possible that are also fast, ideally it should be possible to run it in a jupyter notebook/ colab notebook (but this is not the core thing right now), for now just running in the browser is fine.


#### REFINED SCOPE

# HyperView: Product Scope & Technical Architecture

**Goal:** Build the first "Local-First" Multimodal Data Explorer that natively supports Hierarchical (Hyperbolic) Geometry.
**Target:** A `pip install hyperview` tool 
-----

## 1\. Product Vision: The "Embedded" Philosophy

HyperView rejects the heavy client-server model (Docker containers, separate databases) common in MLOps tools. Instead, it operates as a lightweight, embedded utility.

  * **No Docker:** Runs entirely in the user's Python environment.
  * **No Cloud:** Data stays on the user's local disk.
  * **No Latency:** Uses zero-copy data transfer (Apache Arrow) between disk and visualization.

## 2\. High-Level Architecture

The system consists of a **Python Backend** (the engine) and a **React Frontend** (the viewer). They are packaged together but developed loosely coupled.

### The Stack

| Component | Choice | Justification |
| :--- | :--- | :--- |
| **Database** | **LanceDB** | Embedded, serverless, handles 1M+ vectors on-disk (DiskANN), zero-copy Arrow integration. |
| **Backend** | **FastAPI** | Lightweight async server to handle WebSocket state and serve binary Arrow streams. |
| **Compute** | **Embed-Anything** | Rust-based, ONNX-accelerated inference for generating embeddings locally. |
| **Frontend** | **React + Vite** | Industry standard for SPAs. Vite enables efficient static bundling. |
| **State** | **Zustand** | Handles high-frequency state (mouse movements over 1M points) without triggering React render loops. |
| **Visualization** | **Deck.gl** | The only WebGL library capable of rendering 1M+ interactive points with custom shaders. |

-----

## 3\. Frontend-Backend Integration Strategy

A critical requirement is a seamless developer experience (DX) where the frontend can be iterated on without constant rebuilding, while the final product is a single installable Python package.

### A. Development Mode (Decoupled)

  * **Terminal 1 (Backend):** Run FastAPI on `localhost:8000`. It serves the API endpoints (`/api/points`, `/api/images`).
  * **Terminal 2 (Frontend):** Run Vite Dev Server on `localhost:5173`.
  * **The Bridge:** Configure `vite.config.js` to **proxy** API requests.
    ```javascript
    // vite.config.js
    server: {
      proxy: {
        '/api': 'http://localhost:8000' // Forward API calls to Python
      }
    }
    ```
  * **Benefit:** The frontend developer gets Hot Module Replacement (HMR). Changes reflect instantly. They don't need to restart Python or rebuild the package.

### B. Production Mode (Bundled)

When the package is built for PyPI (`pip install`), the frontend becomes a static asset inside the Python wheel.

1.  **Build:** Run `npm run build` in the frontend folder. Vite compiles everything into a `dist/` directory (HTML, CSS, JS assets).
2.  **Move:** A script copies `dist/` into `hyperview/static/`.
3.  **Package:** `pyproject.toml` (or `MANIFEST.in`) is configured to include `hyperview/static/**/*` as package data.
4.  **Serve:** The FastAPI application detects it is running in production and mounts the static folder:
    ```python
    # main.py
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=pkg_resources.resource_filename("hyperview", "static"), html=True))
    ```

-----

## 4\. Key Features (MVP)

### Feature 1: The "Dual-Geometry" View

The user must see two realities side-by-side to understand the value of hyperbolic space.

  * **Grid View:** Virtualized infinite scroll of images (lazy loaded from disk via API).
  * **Embedding View:** A WebGL scatterplot with a "Geometry Toggle":
      * **Euclidean Mode:** Standard t-SNE/UMAP projection.
      * **Hyperbolic Mode:** Points are mapped onto the Poincaré disk using a custom **Vertex Shader** injected into Deck.gl. This projects the points on the GPU, ensuring 60FPS performance even when toggling geometries.

### Feature 2: The "Hyper-Adapter"

  * **Input:** Standard Euclidean vectors.
  * **Process:** To research further, see current poc implementation.

### Feature 3: Bi-Directional Selection

State is synchronized via Zustand.

  * **Lasso Select:** Drawing a circle on the Poincaré disk filters the Grid View to show only those images.
  * **Image Select:** Clicking a specific image in the Grid highlights its corresponding dot in the Embedding View, allowing users to see where specific samples "live" in the geometry.

-----


TO showcase the demo use a puvblic dataset that would show the strenght of the hyperbolic embeddings compared to euclidean ones, an idea is iNaturalist mini ... research further.