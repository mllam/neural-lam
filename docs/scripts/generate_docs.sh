#!/usr/bin/env bash
# docs/scripts/generate_docs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"

if ! command -v uv &>/dev/null; then
  echo "[ERROR] uv not found. Please install uv first." >&2
  exit 1
fi

echo "[INFO] Syncing dependencies..."
uv pip install -e ".[cpu,docs]"

echo "[INFO] Removing old build at docs/_build..."
rm -rf "$DOCS_DIR/_build"

echo "[INFO] Generating UML architecture diagrams with pyreverse..."
mkdir -p "$DOCS_DIR/_static/uml"
if [ -f "$REPO_ROOT/.venv/bin/pyreverse" ]; then
  (cd "$REPO_ROOT" && "$REPO_ROOT/.venv/bin/pyreverse" -o mmd -p models neural_lam.models -d "$DOCS_DIR/_static/uml/" || echo "[WARN] pyreverse failed, continuing...")
else
  (cd "$REPO_ROOT" && pyreverse -o mmd -p models neural_lam.models -d "$DOCS_DIR/_static/uml/" || echo "[WARN] pyreverse failed, continuing...")
fi

echo "[INFO] Building site with jupyter-book..."
if [ -f "$REPO_ROOT/.venv/bin/jupyter-book" ]; then
  "$REPO_ROOT/.venv/bin/jupyter-book" build docs/ --keep-going
else
  # fallback to uv run if .venv structure differs
  uv run jupyter-book build docs/ --keep-going
fi

INDEX_HTML="$DOCS_DIR/_build/html/index.html"
echo "[OK] Build succeeded! Open: $INDEX_HTML"

if command -v open &>/dev/null; then
  open "$INDEX_HTML"
elif command -v xdg-open &>/dev/null; then
  xdg-open "$INDEX_HTML"
else
  echo "Please open $INDEX_HTML in your browser manually."
fi
