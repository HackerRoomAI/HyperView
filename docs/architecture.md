# HyperView System Architecture

## The "Hybrid Engine" Approach

HyperView is designed to solve the "Representation Collapse" problem at scale (1M+ samples). This requires a hybrid architecture that leverages the best tools for each geometric task:

*   **Python (PyTorch/Geoopt):** For differentiable manifold operations (Training the Adapter).
*   **Rust (Qdrant):** For low-latency retrieval and storage (The "Memory").
*   **WebGL (Deck.gl):** For rendering the Poincaré disk in the browser (The "Lens").

## System Diagram

```mermaid
graph TD
    subgraph "Ingestion (Python)"
        A[Raw Data (Images/Text)] -->|CLIP/ResNet| B[Euclidean Vectors]
        B -->|Hyperbolic Adapter (Geoopt)| C[Hyperbolic Vectors]
    end

    subgraph "Storage & Retrieval (Rust)"
        C -->|gRPC| D[Qdrant Vector DB]
        D -->|Custom Metric| E{Poincaré Distance}
        E -->|HNSW Index| F[Nearest Neighbors]
    end

    subgraph "Visualization (Browser)"
        F -->|JSON/Arrow| G[React Frontend]
        G -->|Deck.gl Shader| H[Poincaré Disk Plot]
        H -->|User Interaction| I[Selection/Curation]
    end
```

## Component Breakdown

### 1. The Hyperbolic Adapter (Python)
*   **Role:** The "Bridge" between the flat world and the curved world.
*   **Tech:** `torch`, `geoopt`.
*   **Function:** Takes standard embeddings (e.g., 512d CLIP vectors) and projects them into the Poincaré ball using the exponential map (`expmap0`). This is where the "expansion" of minority classes happens.

### 2. The Vector Engine (Rust / Qdrant)
*   **Role:** The "Memory" that stores the expanded space.
*   **Tech:** Qdrant (Forked/Extended).
*   **Challenge:** Standard vector DBs only support Dot/Cosine/Euclidean distance.
*   **Solution:** We implement a custom `PoincareDistance` metric in Rust.
    *   Formula: $d(u, v) = \text{arccosh}\left(1 + 2 \frac{\|u-v\|^2}{(1-\|u\|^2)(1-\|v\|^2)}\right)$
    *   This allows us to perform Nearest Neighbor search *respecting the hierarchy*.

### 3. The Visualizer (WebGL)
*   **Role:** The "Lens" that lets humans see the structure.
*   **Tech:** React, Deck.gl, Custom Shaders.
*   **Challenge:** Rendering 1M points in the browser is hard. Rendering them in non-Euclidean geometry is harder.
*   **Solution:** We use a custom WebGL shader that handles the projection. We do *not* project to 2D Euclidean screen coordinates on the CPU. We send the raw hyperbolic coordinates to the GPU, which renders them directly onto the disk.

## Data Flow: The "Fairness" Pipeline

1.  **Ingest:** User uploads a dataset (e.g., Medical Images).
2.  **Embed:** System generates standard embeddings.
3.  **Expand:** The Adapter projects them to Hyperbolic space. "Rare" cases move to the edge.
4.  **Index:** Qdrant stores them.
5.  **Query:** User selects a "Minority" sample.
6.  **Search:** Qdrant calculates Poincaré distance. It finds the *true* semantic neighbors, not just the ones crowded nearby in Euclidean space.
7.  **Visualize:** The UI shows the distinct separation between the minority group and the rare subgroup.
