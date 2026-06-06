# Contributing to Neural-LAM

Thank you for your interest in contributing to Neural-LAM!
We welcome bug reports, bug fixes, documentation improvements, and new features.

## Getting started

1. **Open or find an issue.** Before writing code, open a
   [GitHub issue](https://github.com/mllam/neural-lam/issues) describing what
   you plan to do. If an issue already exists, comment so others know you are
   working on it. New contributors looking for an approachable first task can
   filter by the
   [`good first issue` label](https://github.com/mllam/neural-lam/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).

2. **Read the existing discussion.** Review the full issue thread and any
   linked PRs - someone may have already proposed a solution or identified a
   blocker.

3. **Set up your environment.** Fork the repo via the GitHub UI, then follow
   the [Installing Neural-LAM section of the README](README.md#installing-neural-lam)
   using the `--group dev` flag. Then activate the hooks:

   ```bash
   pre-commit install
   ```

## Code standards

Code quality is enforced automatically by
[pre-commit hooks](.pre-commit-config.yaml) (Black, isort, Flake8, mypy,
Codespell and others). In addition:

- Add **NumPy-style docstrings** and **type annotations** to every public
  function and class.
- Keep new code consistent with the patterns already in the codebase.

## Before you push

Run **both** checks locally - they are the same ones CI will run:

```bash
pre-commit run --all-files
pytest -vv -s --doctest-modules
```

> **Note:** The first test run downloads ~50 MB of example data via
> [pooch](https://www.fatiando.org/pooch/).

## Pull requests

1. Push your branch and open a PR against `main`.
2. Fill in the
   [pull request template](.github/pull_request_template.md) - it contains the
   full checklists for authors, reviewers, and assignees, including the
   CHANGELOG entry format.
3. A maintainer will review your PR. Small, focused PRs are reviewed faster.

## Getting help

- Join the [mllam Slack workspace](https://kutt.to/mllam)
- [Open a GitHub issue](https://github.com/mllam/neural-lam/issues)
