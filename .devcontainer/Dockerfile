# HyperView Development Container
# Multi-platform support: AMD64 (Codespaces) and ARM64 (Apple Silicon)

FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for ML/Image processing
# - build-essential: C/C++ compiler for native extensions
# - libssl-dev, pkg-config: For embed-anything (CLIP embeddings)
# - libjpeg-dev, libpng-dev, libtiff-dev: For Pillow (image processing)
# - libopenblas-dev, liblapack-dev: For NumPy/SciPy acceleration
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libopenblas-dev \
    liblapack-dev \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog

# Set working directory
WORKDIR /workspaces/HyperView

# Create cache directory for uv
RUN mkdir -p /workspaces/HyperView/.cache/uv && \
    chown -R vscode:vscode /workspaces/HyperView/.cache
