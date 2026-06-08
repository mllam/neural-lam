# Contributing to Neural-LAM

Thank you for your interest in contributing to Neural-LAM!
We welcome bug reports, bug fixes, documentation improvements, and new features.

## Community

Neural-LAM is developed in the open by a small, friendly group of researchers
and engineers working on ML-based limited area modelling. We try hard to keep
the community welcoming, open-minded, and constructive - whether you are
opening your first issue or proposing a substantial new feature.

A few things that help keep it that way:

- **Assume good intent.** Reviewers and contributors are mostly volunteering
  their time. If a comment feels blunt, read it as direct rather than hostile.
- **Ask questions early.** It is always cheaper to discuss a design choice
  before code is written. We would rather see a half-formed idea in an issue
  than a finished PR going in the wrong direction.
- **Credit each other.** When your work builds on someone else's PR or issue,
  link it. When you review, thank the author for the time they invested.
- **It is fine to be a beginner.** Several of the maintainers were
  PhD students or first-time open-source contributors not long ago. Nobody
  expects you to know everything about NWP, ML, or the codebase on day one.

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

## Monthly development meetings

The mllam team meets monthly to coordinate roadmaps across the three core
projects (`neural-lam`, `weather-model-graphs`, `mllam-data-prep`), review
proposed changes, and decide what lands in upcoming releases. The meeting is
on the **second Monday of each month, 10:00-11:00 CEST**, on Zoom (the link
is pinned in the
[`#general` channel of the mllam Slack workspace](https://kutt.to/mllam)).

Everyone is welcome - contributors, users, lurkers, and people who are just
curious about ML-based weather forecasting. If you have a feature you would
like to discuss before opening a PR, or a design question that does not fit
neatly into a GitHub thread, the dev meeting is a good venue. It is also a
good place to learn where these tools are heading.

## Getting help

- Join the [mllam Slack workspace](https://kutt.to/mllam) - chat with
  maintainers and other contributors, and find the Zoom link for the next
  dev meeting in `#general`.
- [Open a GitHub issue](https://github.com/mllam/neural-lam/issues) - best
  for anything that benefits from a written, searchable record.
