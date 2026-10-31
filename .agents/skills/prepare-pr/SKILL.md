---
name: prepare-pr
description: Prepare, lint, test, and verify a Pull Request for Neural-LAM following repository contribution guidelines.
---

# Prepare a Pull Request

Follow these steps before opening or updating a Pull Request to comply with repository contribution guidelines.

## 1. Search Before Creating & Scope Limits
Read and follow the stopping rules in `AGENTS.md#ai-specific-rules`:
- Search existing issues and PRs before writing code (`gh issue list`, `gh pr list`).
- Re-read the full thread, maintainer reviews, and Dev Meeting agreements.
- Confine edits strictly to the requested feature or fix; avoid unrequested refactorings.
- Do not push remote branches or open PRs unless explicitly instructed.

## 2. Update the Changelog
Follow `CONTRIBUTING.md#changelog-entries`:
- Add an entry under the appropriate section in `CHANGELOG.md` (`Added`, `Changed`, `Fixed`, or `Maintenance`).
- Format: `- Description [\#NNN](https://github.com/mllam/neural-lam/pull/NNN) @your-handle`
- Use placeholder `[\#PR_NUMBER](https://github.com/mllam/neural-lam/pull/PR_NUMBER)` locally and update it with the assigned number once the PR is opened.

## 3. Run Pre-Commit Hooks
Run code formatting and linters per `CONTRIBUTING.md#before-you-push`.

When working in a Git worktree, use `--git-common-dir` to point to the main repository's `.venv` and avoid re-syncing dependencies:
```bash
UV_PROJECT_ENVIRONMENT="$(git rev-parse --path-format=absolute --git-common-dir)/../.venv" UV_NO_SYNC=1 pre-commit run --all-files
```

In the main checkout:
```bash
uv run pre-commit run --all-files
```

## 4. Run Tests
Follow the test tiers in `CONTRIBUTING.md#before-you-push`:
- **Fast dev feedback**: `CUDA_VISIBLE_DEVICES="" uv run pytest -m "not requires_real_data and not requires_training"`
- **Pre-push requirement**: `uv run pytest tests/` (or skip remote data downloads on CPU runners: `CUDA_VISIBLE_DEVICES="" uv run pytest -m "not requires_real_data"`).

## 5. Review Diff and AI Attribution
- Inspect the full diff (`git diff`) to ensure no stray files or debugging artifacts remain.
- AI attribution is mandatory per `AGENTS.md#commits`: include the `Co-authored-by:` trailer in every commit produced with AI assistance.
