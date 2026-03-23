## Describe your changes
This PR implements the foundational **Sphinx documentation system** for `neural-lam`, as requested in [Issue #61](https://github.com/mllam/neural-lam/issues/61). 

**Key Changes:**
*   **Sphinx Configuration**: Set up `docs/conf.py` with 8 extensions (autodoc, napoleon, intersphinx, myst-parser, etc.) and `sphinx-book-theme`.
*   **API Reference**: Created `.rst` stubs for **8 core modules** to auto-generate documentation from Python docstrings.
*   **User Guide Migration**: Transformed the monolithic README into a structured navigation tree (`getting_started.md`, `user_guide/data.md`, `user_guide/graphs.md`, `training.md`, `evaluation.md`, `contributing.md`).
*   **CI/CD Pipeline**: Added `.github/workflows/docs.yml` to validate documentation on every PR (warnings become errors).
*   **RTD Deployment**: Configured `.readthedocs.yaml` with CPU-only PyTorch for automated staging and production deployment.

**Motivation and Context:**
Currently, code documentation is buried in docstrings and the README has grown too large to navigate easily. This PR provides a searchable, versioned, and hyperlinked documentation website similar to PyTorch Geometric, making the project more accessible to researchers.

**Dependencies:**
Added `docs` dependency group to `pyproject.toml` including:
`sphinx`, `sphinx-book-theme`, `myst-parser`, `sphinx-autodoc-typehints`, `sphinx-copybutton`.

## Issue Link

Closes #61

## Type of change

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [x] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [x] 📖 Documentation (Addition or improvements to documentation)

## Checklist before requesting a review

- [x] My branch is up-to-date with the target branch - if not update your fork with the changes from the target branch (use `pull` with `--rebase` option if possible).
- [x] I have performed a self-review of my code
- [x] For any new/modified functions/classes I have added docstrings that clearly describe its purpose, expected inputs and returned values
- [x] I have placed in-line comments to clarify the intent of any hard-to-understand passages of my code
- [x] I have updated the [README](README.MD) to cover introduced code changes
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] I have given the PR a name that clearly describes the change, written in imperative form ([context](https://www.gitkraken.com/learn/git/best-practices/git-commit-message#using-imperative-verb-form)).
- [x] I have requested a reviewer and an assignee (assignee is responsible for merging). This applies only if you have write access to the repo, otherwise feel free to tag a maintainer to add a reviewer and assignee.

## Checklist for reviewers

Each PR comes with its own improvements and flaws. The reviewer should check the following:
- [ ] the code is readable
- [ ] the code is well tested
- [x] the code is documented (including return types and parameters)
- [ ] the code is easy to maintain
- [x] the build passes without warnings

## Author checklist after completed review

- [x] I have added a line to the CHANGELOG describing this change, in a section
  reflecting type of change (add section where missing):
  - *added*: when you have added new functionality
  - *changed*: when default behaviour of the code has been changed
  - *fixes*: when your contribution fixes a bug
  - *maintenance*: when your contribution is relates to repo maintenance, e.g. CI/CD or documentation
`maintenance: implemented Sphinx documentation system with RTD and CI/CD validation`

## Checklist for assignee

- [x] PR is up to date with the base branch
- [x] the tests pass
- [x] (if the PR is not just maintenance/bugfix) the PR is assigned to the next milestone. If it is not, propose it for a future milestone.
- [x] author has added an entry to the changelog (and designated the change as *added*, *changed*, *fixed* or *maintenance*)
- Once the PR is ready to be merged, squash commits and merge the PR.
