# Development Setup

## Prerequisites

- Python >=3.10
- Git
- uv (recommended) or pip

## Clone and Install

```bash
git clone https://github.com/mllam/neural-lam.git
cd neural-lam
uv sync --extra cpu --group dev --locked
source .venv/bin/activate
```

## Pre-commit Hooks

We use `pre-commit` to ensure code formatting and quality before commits. The hooks include:
- `black` for code formatting.
- `isort` for import sorting.
- `flake8` for linting.
- `mypy` for static type checking.
- `codespell` for spell checking.
- `interrogate` for docstring coverage.

To install and run the pre-commit hooks:

```bash
pre-commit install
uvx pre-commit run --all-files
```

## Running Tests

We use `pytest` for running our test suite. Note that Weights & Biases (W&B) is automatically disabled during tests.

Run all tests:
```bash
pytest -vv -s --doctest-modules
```

Run tests in a single file:
```bash
pytest tests/test_training.py -vv -s
```

Run a single function test:
```bash
pytest tests/test_training.py::test_fn -vv
```

## Building Documentation

We use `jupyter-book` to build the documentation. You can build it using `uv`:

```bash
uv run jb build docs
```

## Project Structure

- `docs/` - Project documentation.
- `neural_lam/` - Main source code directory containing core modules, models, and data logic.
  - `datastore/` - Datastore classes for loading data.
  - `models/` - Core neural network models.
- `tests/` - Unit tests and test data examples.

## Writing Docstrings

We follow the NumPy-style format for docstrings. All functions and classes must have docstrings including Parameters, Returns, Raises, and Tensor shapes where applicable.

Example:
```python
import torch

def process_state(state: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Process the given state tensor by applying a threshold.

    Parameters
    ----------
    state : torch.Tensor
        The input state tensor of shape (batch_size, num_features).
    threshold : float, optional
        The threshold value to apply, by default 0.5.

    Returns
    -------
    torch.Tensor
        The processed state tensor of shape (batch_size, num_features).

    Raises
    ------
    ValueError
        If the state tensor is empty.
    """
    if state.numel() == 0:
        raise ValueError("State tensor cannot be empty.")
    return torch.where(state > threshold, state, torch.zeros_like(state))
```
