# Contributing

## Development Setup

1. Clone the repository
2. Install with development dependencies:
   ```bash
   uv pip install --group dev -e .
   ```

## Pre-commit Hooks

Any push or Pull Request to the main branch triggers pre-commit hooks for
formatting and linting checks. Test locally before pushing:

```bash
pre-commit run --all-files
```

## Running Tests

All tests in the `tests/` directory run automatically via GitHub Actions:

```bash
pytest -vv -s --doctest-modules
```

## Building Documentation

Install documentation dependencies and build:

```bash
pip install --group docs -e .
cd docs
make html
```

The built documentation will be in `docs/_build/html/`.

## Pull Requests

Please use the [PR template](https://github.com/mllam/neural-lam/blob/main/.github/pull_request_template.md)
and follow the instructions there.

## Contact

Join the [mllam Slack channel](https://join.slack.com/t/ml-lam/shared_invite/zt-2t112zvm8-Vt6aBvhX7nYa6Kbj_LkCBQ)
or open a [GitHub issue](https://github.com/mllam/neural-lam/issues).
