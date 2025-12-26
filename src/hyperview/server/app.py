"""FastAPI application for HyperView."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hyperview.core.dataset import Dataset

# Global dataset reference (set by launch())
_current_dataset: Dataset | None = None
_current_session_id: str | None = None


class SelectionRequest(BaseModel):
    """Request model for selection sync."""

    sample_ids: list[str]


class SampleResponse(BaseModel):
    """Response model for a sample."""

    id: str
    filepath: str
    filename: str
    label: str | None
    thumbnail: str | None
    metadata: dict
    embedding_2d: list[float] | None = None
    embedding_2d_hyperbolic: list[float] | None = None


class DatasetResponse(BaseModel):
    """Response model for dataset info."""

    name: str
    num_samples: int
    labels: list[str]
    label_colors: dict[str, str]


class EmbeddingsResponse(BaseModel):
    """Response model for embeddings data."""

    ids: list[str]
    labels: list[str | None]
    euclidean: list[list[float]]
    hyperbolic: list[list[float]]
    label_colors: dict[str, str]


class SimilarSampleResponse(BaseModel):
    """Response model for a similar sample with distance."""

    id: str
    filepath: str
    filename: str
    label: str | None
    thumbnail: str | None
    distance: float
    metadata: dict
    embedding_2d: list[float] | None = None
    embedding_2d_hyperbolic: list[float] | None = None


class SimilaritySearchResponse(BaseModel):
    """Response model for similarity search results."""

    query_id: str
    k: int
    results: list[SimilarSampleResponse]


def create_app(dataset: Dataset | None = None, session_id: str | None = None) -> FastAPI:
    """Create the FastAPI application.

    Args:
        dataset: Optional dataset to serve. If None, uses global dataset.

    Returns:
        FastAPI application instance.
    """
    global _current_dataset, _current_session_id
    if dataset is not None:
        _current_dataset = dataset
    if session_id is not None:
        _current_session_id = session_id

    app = FastAPI(
        title="HyperView",
        description="Dataset visualization with hyperbolic embeddings",
        version="0.1.0",
    )

    # CORS middleware for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/__hyperview__/health")
    async def hyperview_health():
        return {
            "name": "hyperview",
            "version": app.version,
            "session_id": _current_session_id,
            "dataset": _current_dataset.name if _current_dataset is not None else None,
            "pid": os.getpid(),
        }

    @app.get("/api/dataset", response_model=DatasetResponse)
    async def get_dataset_info():
        """Get dataset metadata."""
        if _current_dataset is None:
            raise HTTPException(status_code=404, detail="No dataset loaded")

        return DatasetResponse(
            name=_current_dataset.name,
            num_samples=len(_current_dataset),
            labels=_current_dataset.labels,
            label_colors=_current_dataset.get_label_colors(),
        )

    @app.get("/api/samples")
    async def get_samples(
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        label: str | None = None,
    ):
        """Get paginated samples with thumbnails."""
        if _current_dataset is None:
            raise HTTPException(status_code=404, detail="No dataset loaded")

        # Use storage backend's native pagination (avoids loading all samples)
        samples, total = _current_dataset._storage.get_samples_paginated(
            offset=offset, limit=limit, label=label
        )

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "samples": [s.to_api_dict(include_thumbnail=True) for s in samples],
        }

    @app.get("/api/samples/{sample_id}", response_model=SampleResponse)
    async def get_sample(sample_id: str):
        """Get a single sample by ID."""
        if _current_dataset is None:
            raise HTTPException(status_code=404, detail="No dataset loaded")

        try:
            sample = _current_dataset[sample_id]
            return SampleResponse(**sample.to_api_dict())
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Sample not found: {sample_id}")

    @app.post("/api/samples/batch")
    async def get_samples_batch(request: SelectionRequest):
        """Get multiple samples by their IDs."""
        if _current_dataset is None:
            raise HTTPException(status_code=404, detail="No dataset loaded")

        samples = []
        for sample_id in request.sample_ids:
            try:
                sample = _current_dataset[sample_id]
                samples.append(sample.to_api_dict(include_thumbnail=True))
            except KeyError:
                continue

        return {"samples": samples}

    @app.get("/api/embeddings", response_model=EmbeddingsResponse)
    async def get_embeddings():
        """Get all embeddings for visualization."""
        if _current_dataset is None:
            raise HTTPException(status_code=404, detail="No dataset loaded")

        samples = [
            s
            for s in _current_dataset.samples
            if s.embedding_2d is not None and s.embedding_2d_hyperbolic is not None
        ]

        if not samples:
            raise HTTPException(
                status_code=400, detail="No embeddings computed. Call compute_visualization() first."
            )

        return EmbeddingsResponse(
            ids=[s.id for s in samples],
            labels=[s.label for s in samples],
            euclidean=[s.embedding_2d for s in samples],
            hyperbolic=[s.embedding_2d_hyperbolic for s in samples],
            label_colors=_current_dataset.get_label_colors(),
        )

    @app.post("/api/selection")
    async def sync_selection(request: SelectionRequest):
        """Sync selection state (for future use)."""
        return {"status": "ok", "selected": request.sample_ids}

    @app.get("/api/search/similar/{sample_id}", response_model=SimilaritySearchResponse)
    async def search_similar(
        sample_id: str,
        k: int = Query(10, ge=1, le=100),
        use_hyperbolic: bool = Query(False),
    ):
        """Return k nearest neighbors for a given sample."""
        if _current_dataset is None:
            raise HTTPException(status_code=404, detail="No dataset loaded")

        try:
            similar = _current_dataset.find_similar(
                sample_id, k=k, use_hyperbolic=use_hyperbolic
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Sample not found: {sample_id}")

        results = []
        for sample, distance in similar:
            try:
                thumbnail = sample.get_thumbnail_base64()
            except Exception:
                thumbnail = None

            results.append(
                SimilarSampleResponse(
                    id=sample.id,
                    filepath=sample.filepath,
                    filename=sample.filename,
                    label=sample.label,
                    thumbnail=thumbnail,
                    distance=distance,
                    metadata=sample.metadata,
                    embedding_2d=sample.embedding_2d,
                    embedding_2d_hyperbolic=sample.embedding_2d_hyperbolic,
                )
            )

        return SimilaritySearchResponse(
            query_id=sample_id,
            k=k,
            results=results,
        )

    @app.get("/api/thumbnail/{sample_id}")
    async def get_thumbnail(sample_id: str):
        """Get thumbnail image for a sample."""
        if _current_dataset is None:
            raise HTTPException(status_code=404, detail="No dataset loaded")

        try:
            sample = _current_dataset[sample_id]
            thumbnail_b64 = sample.get_thumbnail_base64()
            return JSONResponse({"thumbnail": thumbnail_b64})
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Sample not found: {sample_id}")

    # Serve static frontend files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    else:
        # Fallback: serve a simple HTML page
        @app.get("/")
        async def root():
            return {"message": "HyperView API", "docs": "/docs"}

    return app


def set_dataset(dataset: Dataset) -> None:
    """Set the global dataset for the server."""
    global _current_dataset
    _current_dataset = dataset
