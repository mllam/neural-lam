# Installation

## Prerequisites
- Python >=3.10
- Git
- (Optional) CUDA-capable GPU for training

## Quick Install with uv (Recommended)

uv is the recommended tool for installing Neural-LAM and its dependencies.

```{code-block} bash
# Clone the repository
git clone https://github.com/mllam/neural-lam.git
cd neural-lam

# CPU-only install
uv sync --extra cpu --group dev --locked

# GPU install (CUDA 13.0, default)
uv sync --extra gpu --group dev --locked

# GPU install (CUDA 12.8)
uv sync --extra gpu-cu128 --group dev --locked
```

```{note}
The extras system (`cpu` / `gpu` / `gpu-cu128`) selects the correct PyTorch index via `[tool.uv.sources]` in `pyproject.toml`.
```

## Install with pip (Alternative)

Alternatively, you can install the dependencies using pip. Make sure to install the correct version of PyTorch for your system before installing the package.

```{code-block} bash
# Install PyTorch (example for CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install neural-lam
pip install -e .[dev]
```

## Verify Installation

```{code-block} bash
# Activate the environment
source .venv/bin/activate

# Verify torch is installed
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Verify neural-lam is importable
python -c "import neural_lam; print('Neural-LAM OK')"
```

## Building Documentation Locally

```{code-block} bash
# Install docs dependencies
uv sync --extra cpu --extra docs --group dev

# Build the documentation
sphinx-build -b html docs/ docs/_build/html/

# Open in browser
open docs/_build/html/index.html  # macOS
# xdg-open docs/_build/html/index.html  # Linux
```

```{seealso}
See the {doc}`quickstart` guide to run your first training.
```
