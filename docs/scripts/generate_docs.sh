#!/usr/bin/env bash
# docs/scripts/generate_docs.sh

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'
info() { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NEURAL_LAM_DIR="$REPO_ROOT/neural_lam"
DOCS_DIR="$REPO_ROOT/docs"

for tool in interrogate pydocstyle jupyter-book; do
  command -v "$tool" &>/dev/null || {
    error "$tool not found. Run: pdm install --group docs"
    exit 1
  }
done

# ── 1. interrogate ──
info "interrogate: docstring coverage audit (fail-under 50)"
interrogate "$NEURAL_LAM_DIR" \
  --fail-under 50 \
  --verbose \
  --generate-badge "$DOCS_DIR/" \
  2>&1 | tee "$DOCS_DIR/interrogate_report.txt" &&
  success "Coverage ≥ 50% ✓" ||
  {
    error "Coverage below 50%! See docs/interrogate_report.txt"
    exit 1
  }

# ── 2. pydocstyle ──
info "pydocstyle: style check (non-blocking)"
pydocstyle "$NEURAL_LAM_DIR" --convention=numpy --add-ignore=D100,D104,D105 &&
  success "pydocstyle passed ✓" ||
  warn "pydocstyle issues (non-blocking)"

# ── 3. jupyter-book build ──
info "jupyter-book: building site"
jupyter-book build "$DOCS_DIR/" &&
  success "Build succeeded ✓" ||
  {
    error "Build FAILED"
    exit 1
  }

echo -e "\n${BOLD} Done — open: $DOCS_DIR/_build/html/index.html${NC}"
