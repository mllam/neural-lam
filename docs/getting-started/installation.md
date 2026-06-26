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
