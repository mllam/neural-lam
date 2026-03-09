# Contributing to Neural-LAM

Thank you for your interest in contributing to neural-lam! This guide covers
everything you need — from setting up your development environment, through
writing and testing code, to opening a pull request.

---

## Table of contents

- [Ways to contribute](#ways-to-contribute)
- [Prerequisites](#prerequisites)
- [Environment setup](#environment-setup)
- [What the pre-commit hooks enforce](#what-the-pre-commit-hooks-enforce)
- [Code standards](#code-standards)
- [Testing](#testing)
- [Before opening a pull request](#before-opening-a-pull-request)
- [Submitting a pull request](#submitting-a-pull-request)
- [Review process](#review-process)
- [CI pipeline overview](#ci-pipeline-overview)
- [Repository structure](#repository-structure)
- [macOS notes](#macos-notes)
- [Troubleshooting](#troubleshooting)
- [Getting help](#getting-help)

---

## Ways to contribute

| Contribution type | Where to start |
|-------------------|---------------|
| 🐛 Bug fixes | Browse [open issues](https://github.com/mllam/neural-lam/issues) labelled `bug` |
| ✨ New features | Check the [roadmap / milestones](https://github.com/mllam/neural-lam/milestones) and open issues |
| 📖 Documentation | Look for issues labelled `documentation` or improve docstrings |
| 🧪 Tests | Increase coverage or add edge-case tests |
| 🔧 Maintenance | CI/CD improvements, dependency updates, linting fixes |

If you want to work on something that doesn't have an issue yet, **open one
first** to discuss the approach before writing code.

---

## Prerequisites

- **Python 3.10 or higher**
- **Git**
- **uv** (recommended) or **pip** — the two package managers used and tested in CI

> **Why `uv`?**
> neural-lam CI runs the full test suite using `uv` and performs a smoke-test
> import check with `pip`. Both are first-class install methods. We recommend
> `uv` because it is fastest and matches the primary CI path.
>
> **Build backend note:** The project uses `pdm-backend` as its build system
> (defined in `pyproject.toml`), but you do **not** need to install PDM to
> develop or contribute. `uv` and `pip` handle everything.

---

## Environment setup

### Step 1 — Fork and clone

```bash
# Fork via GitHub UI, then:
git clone https://github.com/<your-username>/neural-lam.git
cd neural-lam
```

### Step 2 — Install a package manager

**Option A — `uv` (recommended):**

```bash
pip install uv
uv --version
```

Or follow the [official install instructions](https://docs.astral.sh/uv/getting-started/installation).

**Option B — `pip`:** Ships with Python. No extra install needed.

### Step 3 — Install PyTorch

Because PyTorch publishes separate packages for CPU-only and different CUDA
versions, install `torch` **before** installing neural-lam.

```bash
# CPU-only (default for most contributors) — using uv:
uv venv --no-project
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# CPU-only — using pip:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For CUDA GPU support, replace the index URL with the one matching your CUDA
version. Example for CUDA 12.8:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Find the correct URL at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally).
The CI configuration in [`.github/workflows/install-and-test.yml`](../../.github/workflows/install-and-test.yml)
covers all tested variants and can be used as a reference.

### Step 4 — Install neural-lam with dev dependencies

```bash
# Using uv (editable install + dev toolchain):
uv pip install --group dev -e .

# Using pip:
python -m pip install --group dev -e .
```

This installs neural-lam in editable mode along with the full development
toolchain — `pre-commit`, `pytest`, `pooch`, and all runtime dependencies
from `pyproject.toml`.

### Step 5 — Install pre-commit hooks

```bash
pre-commit install
```

This registers the hooks with Git so they run automatically on every commit.
You only need to do this **once per clone**.

> **Why this matters:** Without this step, your commits bypass the formatting
> and linting checks. You will only find out about failures after pushing,
> when CI rejects the PR.

### Step 6 — Verify your setup

Run the full pre-commit suite:

```bash
pre-commit run --all-files
```

All 13 hooks should pass on a clean checkout. If anything fails here, resolve it
before writing code.

Run the test suite:

```bash
pytest -vv -s --doctest-modules
```

This is the same command CI executes. It runs all tests in `tests/` **and**
any doctests embedded in the source modules.

> **First run may take a few minutes.** The test fixtures download example
> data (~50 MB) from an S3 bucket via `pooch` and cache it locally under
> `tests/datastore_examples/`.

### Step 7 — Create a feature branch

```bash
git checkout -b your-feature-name
```

Use a descriptive branch name, for example:
`fix-step-length-calculation`, `add-new-datastore-backend`, `docs-contributing-guide`.

---

## What the pre-commit hooks enforce

Every push and pull request triggers these 13 hooks automatically. Knowing the
rules **before** you write code prevents failed CI runs and review delays.

### Syntax and file hygiene (from `pre-commit-hooks`)

| Hook | What it checks |
|------|---------------|
| `check-ast` | Validates Python syntax — catches invalid Python before anything else runs |
| `check-case-conflict` | Detects filenames that would conflict on case-insensitive file systems (macOS, Windows) |
| `check-docstring-first` | Ensures docstrings appear before any other code in a module |
| `check-symlinks` | Detects dangling symbolic links |
| `check-toml` | Validates TOML files (e.g. `pyproject.toml`) |
| `check-yaml` | Validates YAML configuration files |
| `debug-statements` | Catches accidental `pdb`, `breakpoint()`, `ipdb`, or similar debug calls left in code |
| `end-of-file-fixer` | Ensures every file ends with exactly one newline |
| `trailing-whitespace` | Removes trailing whitespace from all lines |

### Code quality and formatting

| Hook | What it checks |
|------|---------------|
| `codespell` | Spelling errors in comments, docstrings, and variable names |
| `black` | Code formatting — all Python files must conform to Black (line-length: 80) |
| `isort` | Import ordering — imports must be sorted consistently (profile: `black`) |
| `flake8` | Style linting using `Flake8-pyproject` plugin (see note on config below) |
| `mypy` | Static type checking with additional stubs (`types-PyYAML`, `types-Pillow`, `types-tqdm`) |

---

## Code standards

### Formatting rules

| Rule | Tool | Setting |
|------|------|---------|
| Line length | `black`, `isort` | 80 characters |
| Code formatting | `black` | Default settings, line-length 80 |
| Import ordering | `isort` | Profile: `black`; grouped with section headings |
| Linting | `flake8` | With `Flake8-pyproject` plugin (see note below) |
| Type checking | `mypy` | Runs on all Python files |
| Spelling | `codespell` | Comments, docstrings, variable names |

**Flake8 configuration note:** The repo has two flake8 configs that coexist:

| Config file | `max-line-length` | Ignores |
|-------------|-------------------|---------|
| `pyproject.toml` (`[tool.flake8]`) | 80 | `E203`, `I002`, `W503`; `F401` in `__init__.py` |
| `.flake8` (root) | 88 | `E203`, `F811`, `I002`, `W503` |

The standalone `.flake8` file takes precedence when flake8 runs directly.
The `Flake8-pyproject` plugin (used by pre-commit) reads from `pyproject.toml`.
In practice, `black` enforces 80-char lines, so code will conform to the
stricter limit regardless.

### Import conventions

`isort` groups imports into four sections with comment headings:

```python
# Standard library
import os
from pathlib import Path

# Third-party
import torch
import numpy as np

# First-party
from neural_lam.models import GraphLAM

# Local
from .utils import helper_function
```

### Docstrings

Every public class, method, and function should have a docstring that describes
its purpose, expected inputs, and return values:

```python
def create_graph(config_path: str, name: str) -> Graph:
    """Create a graph structure for the given configuration.

    Parameters
    ----------
    config_path : str
        Path to the neural-lam configuration file.
    name : str
        Name identifier for the graph (e.g., 'multiscale', 'hierarchical').

    Returns
    -------
    Graph
        The constructed graph object.
    """
```

### Type annotations

Add type annotations to function signatures. `mypy` runs on every push and
will flag type errors. The project uses these additional type stubs:
`types-PyYAML`, `types-Pillow`, `types-tqdm`.

---

## Testing

### Running tests

```bash
# Full test suite (matches CI):
pytest -vv -s --doctest-modules

# Run a specific test file:
pytest tests/test_training.py -vv -s

# Run a specific test:
pytest tests/test_training.py::test_function_name -vv -s
```

### Test file inventory

| Test file | What it covers |
|-----------|---------------|
| `test_imports.py` | Verifies that `neural_lam` imports successfully |
| `test_cli.py` | Command-line interface smoke tests |
| `test_config.py` | Configuration loading and validation |
| `test_datasets.py` | `WeatherDataset` data loading and sampling |
| `test_datastores.py` | All datastore backends (MDP, NpyFilesMEPS, Dummy) |
| `test_graph_creation.py` | Graph generation for different model types |
| `test_training.py` | End-to-end training loop with tiny datasets |
| `test_plotting.py` | Visualization and plot generation |
| `test_clamping.py` | Output clamping functionality |
| `test_time_slicing.py` | Temporal data slicing and windowing |

### Writing new tests

- Place tests in the `tests/` directory
- Name test files `test_<module>.py` and test functions `test_<what_is_being_tested>`
- Use the existing fixtures in `tests/conftest.py` for datastore examples
- Use `DummyDatastore` from `tests/dummy_datastore.py` when possible to avoid
  downloading external data
- Some tests download example datasets on first run (~50 MB MEPS data from S3
  via `pooch`); first runs may take a few minutes

---

## Before opening a pull request

Run **both** of these commands and confirm both pass:

```bash
pre-commit run --all-files         # All 13 hooks must pass
pytest -vv -s --doctest-modules    # Full test suite must pass
```

| Check | Command | What it validates |
|-------|---------|-------------------|
| Linting + formatting | `pre-commit run --all-files` | All 13 hooks pass |
| Tests + doctests | `pytest -vv -s --doctest-modules` | Full test suite passes |

> **Any PR that fails either check will be automatically blocked by CI.**
> Catching issues locally before pushing avoids failed CI runs and unnecessary
> review round-trips.

---

## Submitting a pull request

1. Push your branch to your fork
2. Open a pull request against `main`
3. Fill in the [PR template](../../.github/pull_request_template.md) completely
4. Select the correct type of change (bug fix, new feature, breaking change,
   or documentation)

### PR checklist

Before requesting review, confirm:

- [ ] Branch is up to date with `main` (rebase if needed)
- [ ] Self-review completed
- [ ] Docstrings added for new/modified public functions and classes
- [ ] In-line comments for any hard-to-understand logic
- [ ] README updated if the change affects usage
- [ ] Tests added for bug fixes or new features
- [ ] PR title is descriptive and uses
  [imperative form](https://www.gitkraken.com/learn/git/best-practices/git-commit-message#using-imperative-verb-form)
  (e.g., "Add clamping support for output features")
- [ ] Reviewer and assignee requested (if you have write access; otherwise tag
  a maintainer)

### Naming your PR

- `Add support for MLFlow logging`
- `Fix step length calculation for dt >= 24h`
-  `Fixed some stuff`
- `WIP changes`

---

## Review process

### What reviewers look for

- Code is readable and well-structured
- Code is well tested (new functionality has tests)
- Code is documented (docstrings with parameter types and return values)
- Code is easy to maintain

### After review approval

1. Add an entry to [`CHANGELOG.md`](../../CHANGELOG.md) under the `[unreleased]`
   section, in the appropriate category:
   - **Added** — new functionality
   - **Changed** — default behaviour changes
   - **Fixed** — bug fixes
   - **Maintenance** — CI/CD, documentation, dependencies

   Format:
   ```markdown
   - Short description of change [\#123](https://github.com/mllam/neural-lam/pull/123) @yourusername
   ```

2. Squash commits if requested
3. The assignee merges the PR once all checks pass

---

## CI pipeline overview

When you push or open a PR, three GitHub Actions workflows run automatically:

| Workflow | File | What it does |
|----------|------|-------------|
| **Linting** | `pre-commit.yml` | Runs all 13 pre-commit hooks across Python 3.10 – 3.14 |
| **CPU+GPU testing** | `install-and-test.yml` | Installs with `pip` and `uv`, runs tests on CPU (ubuntu) and GPU (AWS via cirun.io) using Python 3.13 |
| **Package release** | `ci-pypi-deploy.yml` | Builds the wheel and uploads to PyPI on tagged releases |

Linting and testing must both pass before a PR can be merged. The package
release workflow runs on pushes to `main` and published releases but does not
block PRs.

---

## Repository structure

```
neural-lam/
├── neural_lam/                    # Main source package
│   ├── __init__.py                # Package init, version detection
│   ├── config.py                  # Configuration dataclasses
│   ├── create_graph.py            # Graph generation CLI
│   ├── train_model.py             # Training / evaluation CLI
│   ├── weather_dataset.py         # PyTorch Dataset for weather data
│   ├── interaction_net.py         # Interaction network layers
│   ├── loss_weighting.py          # Loss weighting strategies
│   ├── metrics.py                 # Evaluation metrics (RMSE, etc.)
│   ├── utils.py                   # Shared utilities
│   ├── vis.py                     # Visualization helpers
│   ├── custom_loggers.py          # W&B / MLFlow logger wrappers
│   ├── plot_graph.py              # Graph plotting utilities
│   ├── models/                    # Model classes
│   │   ├── __init__.py
│   │   ├── ar_model.py            # Autoregressive base model
│   │   ├── base_graph_model.py    # Base class for graph models
│   │   ├── base_hi_graph_model.py # Base class for hierarchical models
│   │   ├── graph_lam.py           # GraphLAM (1-level and multiscale)
│   │   ├── hi_lam.py              # HiLAM (sequential hierarchical)
│   │   └── hi_lam_parallel.py     # HiLAM-Parallel
│   └── datastore/                 # Datastore backends
│       ├── __init__.py            # Registry and init_datastore()
│       ├── base.py                # BaseDatastore abstract class
│       ├── mdp.py                 # MDPDatastore (mllam-data-prep / zarr)
│       ├── npyfilesmeps/          # NpyFilesDatastoreMEPS (numpy-based)
│       └── plot_example.py        # Datastore plotting utility
│
├── tests/                         # Test suite
│   ├── conftest.py                # Shared fixtures and test data setup
│   ├── dummy_datastore.py         # In-memory datastore for fast tests
│   ├── datastore_examples/        # Example configs and data for tests
│   │   ├── mdp/                   # MDP (DANRA) test data
│   │   └── npyfilesmeps/          # NpyFiles MEPS test data (downloaded)
│   ├── test_imports.py
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_datasets.py
│   ├── test_datastores.py
│   ├── test_graph_creation.py
│   ├── test_training.py
│   ├── test_plotting.py
│   ├── test_clamping.py
│   └── test_time_slicing.py
│
├── docs/
│   ├── contributing/              # This guide
│   └── notebooks/                 # Jupyter notebooks for analysis
│
├── figures/                       # Images used in README
│
├── .github/
│   ├── workflows/
│   │   ├── pre-commit.yml         # Linting CI
│   │   ├── install-and-test.yml   # CPU+GPU testing CI
│   │   └── ci-pypi-deploy.yml     # PyPI release CI
│   └── pull_request_template.md
│
├── pyproject.toml                 # Project metadata, deps, tool config
├── .pre-commit-config.yaml        # Pre-commit hook definitions
├── .flake8                        # Flake8 standalone config (overrides)
├── .cirun.yml                     # AWS GPU runner config for CI
├── .gitignore
├── CHANGELOG.md
├── LICENSE.txt
└── README.md
```

> **Note:** `pdm.lock` and `.pdm-python` exist locally but are listed in
> `.gitignore` and are not tracked in version control.

---

## macOS notes

- `uv` and `pip` install cleanly on both Intel and Apple Silicon Macs.
- The CPU-only `torch` variant works without issues on Apple Silicon.
- CI lints across Python **3.10 – 3.14** and runs the test suite on **3.13**.
  If you are using a different version locally, be aware that some hooks (e.g.
  `mypy`) may produce different results.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `pre-commit` not found | Make sure you installed with `--group dev` and your virtual environment is activated |
| `mypy` errors on clean checkout | Try running with the same Python version CI uses (3.13) |
| Tests fail downloading data | Check your internet connection; data is fetched from `https://object-store.os-api.cci1.ecmwf.int` |
| `torch` import errors | Ensure you installed `torch` **before** installing neural-lam (Step 3 before Step 4) |
| `pooch` not found | You likely installed without `--group dev`; re-run Step 4 |

---

## Getting help

- **Slack:** Join the [mllam Slack workspace](https://kutt.to/mllam) (request
  access after following the link)
- **Issues:** [Open an issue](https://github.com/mllam/neural-lam/issues) on
  GitHub for bugs, questions, or feature discussions
- **Maintainers:** Tag `@joeloskarsson` or `@leifdenby` for guidance on larger
  contributions
