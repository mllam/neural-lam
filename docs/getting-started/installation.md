# Installation

## Prerequisites
- Python >=3.10
- Git
- (Optional) CUDA-capable GPU for training

When installing `neural-lam` you have a choice of either installing directly with `pip` or using the `uv` package manager.
We recommend using `uv` as it makes it easy to add/remove packages while
keeping versions consistent (it automatically updates the `pyproject.toml`
file), makes it easy to handle virtual environments and includes the
development toolchain packages installation too.

**regarding `torch` installation**: `torch` is a required dependency, but it ships separate variants for CPU-only and each CUDA version. With `uv`, the correct PyTorch index is chosen automatically via `[tool.uv.sources]`. With `pip`, you must pass `--index-url` explicitly (see instructions below).

We cover all the installation options in our [github actions ci/cd
setup](https://github.com/mllam/neural-lam/tree/main/.github/workflows) which you can use as a reference.

### From pypi.org

```
python -m pip install neural_lam
```

### From source

#### Using `uv`

1. Clone this repository and navigate to the root directory.
2. Install `uv` if you don't have it installed on your system (either with `pip install uv` or [following the install instructions](https://docs.astral.sh/uv/getting-started/installation)).
3. Install with the CPU or GPU extra:
   ```bash
   uv sync --extra cpu        --group dev  # CPU-only
   uv sync --extra gpu        --group dev  # GPU, CUDA 13.0 (default)
   uv sync --extra gpu-cu128  --group dev  # GPU, CUDA 12.8
   ```
   This creates a virtual environment, installs `torch` from the correct
   PyTorch index, and installs all other dependencies (including dev tools)
   in one step.

   If you need a different CUDA version, install with the `cpu` extra first (smaller download) and then swap `torch` to your CUDA variant:
   ```bash
   uv sync --extra cpu --group dev              # create env (small torch wheel)
   uv pip install torch --torch-backend auto    # replace torch with your CUDA variant
   ```
   Note: this swap is venv-local and will be reverted by the next `uv sync`.

#### Using `pip`

1. Clone this repository and navigate to the root directory.
> If you are happy using the latest version of `torch` with GPU support (expecting the latest version of CUDA is installed on your system) you can skip to step 3.
2. Install a specific version of `torch` with `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` for a CPU-only version or `python -m pip install torch --index-url https://download.pytorch.org/whl/cu130` for CUDA 13.0 support (use `.../whl/cu128` for CUDA 12.8; you can find the correct URL for the variant you want on [PyTorch webpage](https://pytorch.org/get-started/locally/)).
3. Install the dependencies with `python -m pip install .`. If you will be developing `neural-lam` we recommend to install in editable mode and install the development dependencies with `python -m pip install --group dev -e .` so you can make changes to the code and see the effects immediately.

## Building Documentation Locally

```{code-block} bash
# Install docs dependencies (swap cpu for the torch extra you use)
uv sync --extra cpu --extra docs --group dev --locked
source .venv/bin/activate

# Export notebook source to Markdown (does not execute the notebooks)
python docs/export_notebooks.py

# Build the documentation
sphinx-build -W --keep-going -b html docs/ docs/_build/html/

# Open in browser
open docs/_build/html/index.html  # macOS
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
