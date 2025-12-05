#!/bin/bash
# Export frontend to static files for Python package

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
STATIC_DIR="$PROJECT_ROOT/src/hyperview/server/static"

echo "🔄 Building frontend..."
cd "$FRONTEND_DIR"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Build for static export
npm run build

# Copy to Python package
echo "📁 Copying to Python package..."
rm -rf "$STATIC_DIR"
mkdir -p "$STATIC_DIR"
cp -r out/* "$STATIC_DIR/"

echo "✅ Frontend exported to $STATIC_DIR"
echo ""
echo "To test, run:"
echo "  cd $PROJECT_ROOT"
echo "  uv run hyperview demo"
