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
  function and class. The
  [numpydoc style guide](https://numpydoc.readthedocs.io/en/latest/format.html)
  is the authoritative reference;
  [`neural_lam/datastore/base.py`](neural_lam/datastore/base.py) is a good
  in-repo example of the conventions we follow (sectioned `Parameters` /
  `Returns`, types in the signature line, prose first).
- When annotating tensor shapes, use the canonical dimension names from the
  [Dimension Glossary](README.md#dimension-glossary) so shapes stay consistent
  across the codebase.
- Keep new code consistent with the patterns already in the codebase.

## Before you push

Run **both** checks locally - they are the same ones CI will run:

```bash
pre-commit run --all-files
pytest -vv -s --doctest-modules
```

> **Note:** The first test run downloads ~50 MB of example data via
> [pooch](https://www.fatiando.org/pooch/).

## Community roadmap

Our community roadmap is defined by
[milestones](https://github.com/mllam/neural-lam/milestones) on the
[`neural-lam` GitHub repo](https://github.com/mllam/neural-lam). We use
[semantic versioning](https://semver.org/) `vX.Y.Z` for the milestones, so
issues on the roadmap carry the version they will be part of as their
milestone.

The process for putting issues or PRs on the roadmap is:

1. **Propose** - label your issue or PR with a milestone in the form
   `vX.Y.Z (proposed)` (e.g. `v0.9.2 (proposed)`).
2. **Discuss** - at a development meeting the assignee explains the purpose
   and how it fits the current roadmap. The group decides whether to accept
   it and which revision it targets (it may be moved to a later release).
3. **Accept** - once accepted the `(proposed)` suffix is removed and the
   issue or PR is placed on the milestone for that version.

To propose something for the roadmap, all you need to do is add a milestone
label in the form `vX.Y.Z (proposed)`.

## Pull requests

1. Push your branch and open a PR against `main`.
2. Fill in the
   [pull request template](.github/pull_request_template.md) - it contains the
   full checklists for authors, reviewers, and assignees.
3. Write commit messages in **imperative form** matching the existing
   `git log` style ("Add X" not "Added X"), and keep one concern per PR.
4. PRs land via **squash-and-merge**: the PR title becomes the single commit
   message on `main` and the PR description becomes its body. Polish both
   before requesting review - per-commit history on your branch is not
   preserved in `main`.
5. A maintainer will review your PR. Small, focused PRs are reviewed faster.
6. After review, iterate on the feedback. Once the review is resolved and
   CI is green:
   - **Bugfixes and maintenance PRs** are merged directly by the assignee.
   - **Feature PRs** (anything labelled `enhancement`) are proposed for the
     [community roadmap](#community-roadmap) by adding a `vX.Y.Z (proposed)`
     milestone, then discussed at the next
     [monthly dev meeting](#monthly-development-meetings) so the team can
     align on roadmap and scope. Feature PRs accepted for the roadmap
     (i.e. assigned a `vX.Y.Z` milestone without `(proposed)`) are merged
     by the assignee once the milestone is ready.
7. **Accepted feature PRs** are merged when the milestone closes (or
   sooner if the feature is self-contained and the maintainers agree).
   Bugfix and maintenance PRs are merged as soon as step 6 completes.

## CHANGELOG entries

**Every PR must add a line to [CHANGELOG.md](CHANGELOG.md)** under the
section matching the change type (`Added`, `Changed`, `Fixed`, or
`Maintenance`). Add a new section heading if it does not already exist
under the current `[unreleased]` block.

The entry references the **PR number, not the issue number**, and tags the
author. Format:

```markdown
- Short description of the change [\#NNN](https://github.com/mllam/neural-lam/pull/NNN) @your-handle
```

Example (from a real merged PR):

```markdown
- Add bounds checking for `--var_leads_metrics_watch` indices to fail at
  CLI parse time rather than mid-validation [\#306](https://github.com/mllam/neural-lam/pull/306) @your-handle
```

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

## License

Neural-LAM is released under the [MIT License](LICENSE.txt). By contributing
to this repository, you agree that your contribution is licensed under the
same terms.
