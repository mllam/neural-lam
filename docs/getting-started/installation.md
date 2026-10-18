# Installation

## Prerequisites
- Python >=3.10
- Git
- (Optional) CUDA-capable GPU for training

```{include} ../../README.md
:start-after: "# Installing Neural-LAM"
:end-before: "# Using Neural-LAM"
```

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
