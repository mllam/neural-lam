# Contributing to neural-lam

Thank you for your interest in contributing to neural-lam! This document
provides guidelines to help you get started.

## Setting Up the Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
   git clone https://github.com/YOUR_USERNAME/neural-lam.git
   cd neural-lam
```
3. Install dependencies using `uv`:
```bash
   uv sync --group dev
```
4. Install pre-commit hooks:
```bash
   pre-commit install
```

## Running Tests

Run the full test suite locally before submitting a PR:
```bash
pytest tests/
```

## Running Pre-commit Hooks

To check your code before committing:
```bash
pre-commit run --all-files
```

This will run `black`, `isort`, `flake8`, `mypy` and `codespell` on your code.

## Branch Naming Conventions

Please use the following prefixes for your branches:

| Prefix | Use for |
|--------|---------|
| `feat/` | New features or enhancements |
| `fix/` | Bug fixes |
| `docs/` | Documentation changes |
| `test/` | Adding or updating tests |
| `maintenance/` | CI/CD, dependencies, tooling |

Example: `feat/add-type-hints-metrics`

## Pull Request Checklist

Before opening a PR, make sure you have:

- [ ] Branched off from the latest `main`
- [ ] Written or updated tests for your changes
- [ ] Added or updated docstrings for any new/modified functions
- [ ] Run `pre-commit run --all-files` with no errors
- [ ] Added an entry to `CHANGELOG.md`
- [ ] Linked the relevant issue in the PR description

## Commit Message Style

Use the imperative verb form for commit messages:

- ✅ `Add type hints to metrics.py`
- ✅ `Fix missing mask in loss computation`
- ❌ `Added type hints`
- ❌ `Fixing bug`

## Code Style

This project uses:
- `black` for formatting (line length: 80)
- `isort` for import sorting
- `flake8` for linting
- `mypy` for type checking

All of these are enforced via pre-commit hooks.

## Getting Help

- Open a [GitHub Issue](https://github.com/mllam/neural-lam/issues)
- Join the MLLAM Slack: https://kutt.it/mllam
