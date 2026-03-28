#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_DIR="$SCRIPT_DIR/docs"

echo "Neural-LAM Documentation Builder"
echo "=================================="

if [ ! -d "$DOCS_DIR" ]; then
    echo "Error: docs directory not found at $DOCS_DIR"
    exit 1
fi

echo "Checking for Sphinx installation..."
if ! command -v sphinx-build &> /dev/null; then
    echo "Sphinx not found. Installing documentation dependencies..."
    pip install -e ".[docs]"
fi

echo ""
echo "Building HTML documentation..."
cd "$DOCS_DIR"
make clean
make html

echo ""
echo "✓ Documentation built successfully!"
echo "Open the documentation in your browser:"
echo "  file://$DOCS_DIR/_build/html/index.html"
echo ""
echo "Or start a local server:"
echo "  cd $DOCS_DIR/_build/html && python -m http.server 8000"
echo ""
echo "Then visit: http://localhost:8000"
