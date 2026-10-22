---
name: prepare-pr
description: Prepare, lint, test, and verify a Pull Request for Neural-LAM following repository contribution guidelines. Use prior to opening or updating PRs.
---

# Prepare a Pull Request

Follow these steps before opening or updating a Pull Request to comply with `AGENTS.md` and `CONTRIBUTING.md`.

## 1. Search Before Creating & Strict Scope Limits
Duplicate issues and PRs from AI agents are a recurring issue. Apply these stopping rules before writing code:
1. **Search for existing work**:
   ```bash
   gh issue list --state all --search "<keywords>"
   gh pr list --state all --search "<keywords>"
   ```
   If a matching issue or PR already exists, contribute or coordinate there rather than opening a competing one.
2. **Re-read the thread**:
   Read the full issue/PR discussion, including rejected approaches and maintainer feedback, before planning changes.
3. **Limit scope**:
   Confine edits strictly to the requested feature or fix. Do not introduce unrequested refactorings or speculative abstractions. If requirements or target interfaces are ambiguous, ask one clarifying question instead of guessing.
4. **Do not push without a request**:
   Never create remote branches or open pull requests unless explicitly instructed by the user.

## 2. Update the Changelog
Add an entry describing your change under the appropriate section in `CHANGELOG.md`:
- `Added` for new features, models, datastores, or metrics.
- `Changed` for modifications to existing behavior.
- `Fixed` for bug fixes.
- `Maintenance` for internal tooling, documentation, and dependencies.

Format (from `CONTRIBUTING.md`):
```markdown
- Short description of the change [\#NNN](https://github.com/mllam/neural-lam/pull/NNN) @your-handle
```

> **Note on PR Numbers:** Before opening the PR, the PR number is not yet known. Use a placeholder like `[\#PR_NUMBER](https://github.com/mllam/neural-lam/pull/PR_NUMBER)` and `@your-handle`. Once the PR is created and assigned a number on GitHub, update the changelog entry with the actual PR number and your handle before requesting review.

## 3. Run Pre-Commit Hooks
Run code formatting and linters. When running from a git worktree, set environment variables to prevent re-syncing the full 5 GB virtual environment:

```bash
UV_PROJECT_ENVIRONMENT="$(git rev-parse --show-toplevel)/.venv" UV_NO_SYNC=1 pre-commit run --all-files
```

If working in the main repository checkout:

```bash
uv run pre-commit run --all-files
```

## 4. Run Tests
Follow the two-phase testing workflow:
1. **Fast development feedback**:
   Run unit tests that skip external dataset downloads and heavy multi-epoch training:
   ```bash
   CUDA_VISIBLE_DEVICES="" uv run pytest -m "not requires_real_data and not requires_training"
   ```
2. **Pre-PR requirement**:
   Before submitting a PR, run the test suite as specified in `CONTRIBUTING.md` and `AGENTS.md`:
   ```bash
   uv run pytest tests/
   ```
   *(If your environment lacks GPU hardware or external data credentials, run `CUDA_VISIBLE_DEVICES="" uv run pytest -m "not requires_real_data"` and document this in the PR description).*

## 5. Review Diff and Commit Attribution
1. Review the full diff as an independent reviewer to ensure no stray files or debugging artifacts remain:
   ```bash
   git diff --stat
   git diff
   ```
2. AI attribution is mandatory. Ensure every commit created with AI assistance includes the `Co-authored-by:` trailer:
   ```bash
   git commit -m "feat: Add new metric" -m "Co-authored-by: Claude Opus 4.8 <noreply@anthropic.com>"
   ```
