#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

readonly ROOT_DIR
readonly VENV_DIR="$ROOT_DIR/.venv"
readonly VENV_PY="$VENV_DIR/bin/python"
readonly EXAMPLE_CONFIG="tests/datastore_examples/mdp/danra_100m_winds/config.yaml"
readonly EXAMPLE_GRAPH_DIR="tests/datastore_examples/mdp/danra_100m_winds/graph/1level"

export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"
export XDG_CACHE_HOME="$ROOT_DIR/.cache"
export WANDB_MODE=disabled
export WANDB_DIR="$ROOT_DIR/.wandb"

mkdir -p "$MPLCONFIGDIR" "$ROOT_DIR/.cache/fontconfig" "$WANDB_DIR"

pick_python() {
    local candidate=""
    for candidate in "${PYTHON_BIN:-}" python3.10 /opt/homebrew/bin/python3.10 python3.11 python3.12 python3; do
        if [[ -z "$candidate" ]]; then
            continue
        fi
        if ! command -v "$candidate" >/dev/null 2>&1; then
            continue
        fi
        if "$candidate" -c 'import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)' >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    echo "No supported Python interpreter found. Set PYTHON_BIN to Python 3.10-3.12." >&2
    exit 1
}

ensure_venv() {
    if [[ -x "$VENV_PY" ]]; then
        return 0
    fi

    local python_bin
    python_bin="$(pick_python)"
    echo "Creating virtual environment with $python_bin"
    "$python_bin" -m venv "$VENV_DIR"
}

ensure_installed() {
    ensure_venv

    if "$VENV_PY" -c "import neural_lam, torch" >/dev/null 2>&1; then
        return 0
    fi

    local torch_version
    echo "Resolving project PyTorch version"
    torch_version="$("$VENV_PY" -m pip install --dry-run . 2>/dev/null | sed -n 's/.*torch-\([0-9.]*\).*/\1/p' | tail -n 1)"
    if [[ -z "$torch_version" ]]; then
        echo "Failed to resolve a torch version from the project metadata." >&2
        exit 1
    fi

    echo "Installing torch==$torch_version"
    "$VENV_PY" -m pip install "torch==$torch_version"

    echo "Installing project dependencies"
    "$VENV_PY" -m pip install --group dev .
}

run_graph() {
    ensure_installed
    "$VENV_PY" -m neural_lam.create_graph \
        --config_path "$EXAMPLE_CONFIG" \
        --name 1level \
        --levels 1
}

run_train() {
    ensure_installed
    if [[ ! -d "$EXAMPLE_GRAPH_DIR" ]]; then
        run_graph
    fi

    "$VENV_PY" -m neural_lam.train_model \
        --config_path "$EXAMPLE_CONFIG" \
        --model graph_lam \
        --graph 1level \
        --epochs 1 \
        --batch_size 2 \
        --num_workers 0 \
        --hidden_dim 4 \
        --hidden_layers 1 \
        --processor_layers 2 \
        --ar_steps_train 1 \
        --ar_steps_eval 2 \
        --val_steps_to_log 1 2
}

usage() {
    cat <<'EOF'
Usage: scripts/run_mdp_example.sh <setup|graph|train>

  setup  Create the venv and install dependencies
  graph  Generate the example 1-level graph
  train  Run the example 1-epoch training command
EOF
}

main() {
    local action="${1:-}"
    case "$action" in
        setup)
            ensure_installed
            ;;
        graph)
            run_graph
            ;;
        train)
            run_train
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

main "$@"
