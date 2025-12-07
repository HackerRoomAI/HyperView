#!/bin/bash
# Post-create command for HyperView devcontainer
# This script runs once after the container is created

set -e

echo ""
echo "🚀 Setting up HyperView development environment..."
echo ""

# Install uv (platform-agnostic installer)
echo "📦 Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
fi

# Verify uv installation
uv --version

# Create Python virtual environment
echo ""
echo "🐍 Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Add automatic venv activation to bashrc
if ! grep -q "source /workspaces/HyperView/.venv/bin/activate" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Auto-activate HyperView virtual environment" >> ~/.bashrc
    echo "if [ -f /workspaces/HyperView/.venv/bin/activate ]; then" >> ~/.bashrc
    echo "    source /workspaces/HyperView/.venv/bin/activate" >> ~/.bashrc
    echo "fi" >> ~/.bashrc
fi

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies (this may take a few minutes)..."
uv pip install -e ".[dev]"

# Install pre-commit hooks
echo ""
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# Install frontend dependencies
echo ""
echo "📦 Installing frontend dependencies (this may take a few minutes)..."
cd frontend
npm install
cd ..

# Verify installations
echo ""
echo "✅ Verification:"
echo "   Python: $(python --version)"
echo "   uv: $(uv --version)"
echo "   Node.js: $(node --version)"
echo "   npm: $(npm --version)"
echo ""

# Display helpful information
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✨ HyperView development environment ready!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Quick start commands:"
echo ""
echo "  🚀 Run demo (500 CIFAR-100 samples):"
echo "     hyperview demo --samples 500"
echo ""
echo "  🔧 Development mode (backend + frontend):"
echo "     Terminal 1: python scripts/demo.py --samples 200 --no-browser"
echo "     Terminal 2: cd frontend && npm run dev"
echo ""
echo "  🧪 Run tests:"
echo "     pytest"
echo ""
echo "  🎨 Code quality:"
echo "     ruff check src/"
echo "     ruff format src/"
echo "     pre-commit run --all-files"
echo ""
echo "  📦 Export frontend for production:"
echo "     ./scripts/export_frontend.sh"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
