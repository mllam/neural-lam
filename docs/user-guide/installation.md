# Installation

## Prerequisites

- Python >=3.10
- Git
- (Optional) CUDA-capable GPU for training

## From PyPI

```bash
python -m pip install neural_lam
```

## From source using `uv` (recommended)

`uv` manages virtual environments and selects the correct PyTorch index automatically.

1. Clone the repository and navigate to the root directory.
2. Install `uv` if needed (`pip install uv` or follow the [install instructions](https://docs.astral.sh/uv/getting-started/installation)).
3. Install with the CPU or GPU extra:

   ```bash
   uv sync --extra cpu        --group dev  # CPU-only
   uv sync --extra gpu        --group dev  # GPU, CUDA 13.0 (default)
   uv sync --extra gpu-cu128  --group dev  # GPU, CUDA 12.8
   ```

   If you need a different CUDA version, install with the `cpu` extra first and then swap `torch`:

   ```bash
   uv sync --extra cpu --group dev              # create env (small torch wheel)
   uv pip install torch --torch-backend auto    # replace torch with your CUDA variant
   ```

   Note: this swap is venv-local and will be reverted by the next `uv sync`.

## From source using `pip`

1. Clone the repository and navigate to the root directory.
2. (Optional) Install a specific `torch` variant:
   - CPU-only: `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`
   - CUDA 13.0: `python -m pip install torch --index-url https://download.pytorch.org/whl/cu130`
   - CUDA 12.8: `python -m pip install torch --index-url https://download.pytorch.org/whl/cu128`
   - Find other variants on the [PyTorch website](https://pytorch.org/get-started/locally/).
3. Install the package:
   ```bash
   python -m pip install --group dev -e .
   ```

The full CI/CD setup in [`.github/workflows`](https://github.com/mllam/neural-lam/tree/main/.github/workflows) can be used as a reference for all install options.

## Building documentation locally

```bash
# Install docs dependencies
uv sync --extra cpu --extra docs --group dev --locked
source .venv/bin/activate

# Export notebook source to Markdown (does not execute the notebooks)
python docs/export_notebooks.py

# Build the documentation
sphinx-build -W --keep-going -b html docs/ docs/_build/html/

# Open in browser
open docs/_build/html/index.html       # macOS
# xdg-open docs/_build/html/index.html  # Linux
```

Run these commands from the repository root. The exported notebook Markdown
is generated locally and should not be committed. The documentation build
does not need the MEPS dataset; running the dataset-reduction notebook does.
See [Contributing](https://github.com/mllam/neural-lam/blob/main/CONTRIBUTING.md)
for notebook editing and execution instructions.

```{seealso}
See the {doc}`quickstart` guide to run your first training.
```
