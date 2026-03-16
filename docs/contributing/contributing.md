# Contributing to Neural-LAM

Thank you for your interest in contributing! If your idea doesn't have an
issue yet, please [open one](https://github.com/mllam/neural-lam/issues)
first to discuss the approach.

---

## Environment setup

**Requirements:** Python ≥ 3.10, Git,
[uv](https://docs.astral.sh/uv/getting-started/installation) (recommended)
or pip.

```bash
# Fork via GitHub UI, then:
git clone https://github.com/<your-username>/neural-lam.git
cd neural-lam

# Install PyTorch first (CPU example — for GPU see pytorch.org/get-started):
uv venv --no-project && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install neural-lam + dev tools, then set up pre-commit hooks:
uv pip install --group dev -e .
pre-commit install

# Verify:
pre-commit run --all-files
pytest -vv -s --doctest-modules
```

> First test run downloads ~50 MB of example data via `pooch`.

---

## Code standards

Standards are enforced automatically by **pre-commit hooks** (see
[`.pre-commit-config.yaml`](../../.pre-commit-config.yaml) for the full
list). In short: [Black](https://black.readthedocs.io/) formatting (80-char
lines), [isort](https://pycqa.github.io/isort/) imports, Flake8 linting,
mypy type checking, and Codespell.

Add **docstrings** (NumPy style) and **type annotations** to all public
functions and classes.

---

## Testing

```bash
pytest -vv -s --doctest-modules              # full suite (same as CI)
pytest tests/test_training.py -vv -s         # single file
pytest tests/test_training.py::test_fn -vv -s # single test
```

Place new tests in `tests/` as `test_<module>.py`. Prefer `DummyDatastore`
(in `tests/dummy_datastore.py`) over external data downloads.

---

## Pull requests

1. Confirm both `pre-commit run --all-files` and `pytest` pass locally
2. Push your branch and open a PR against `main`
3. Fill in the [PR template](../../.github/pull_request_template.md) — it has
   the full checklists for authors, reviewers, and assignees

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `pre-commit` / `pooch` not found | Reinstall with `--group dev`; activate venv |
| `torch` import errors | Install torch **before** neural-lam |
| Test data download fails | Check internet (`object-store.os-api.cci1.ecmwf.int`) |
| `mypy` mismatches | Use Python 3.13 (same as CI) |

---

## Getting help

[mllam Slack](https://kutt.to/mllam) · [Open an issue](https://github.com/mllam/neural-lam/issues) · Tag `@joeloskarsson` or `@leifdenby` for larger contributions
