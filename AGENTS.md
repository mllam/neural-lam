# AGENTS Guide

This file defines basic rules for AI coding agents contributing to neural-lam.

## Scope
- Keep PRs small and focused on one issue.
- Do not refactor unrelated code.
- Do not change public APIs unless the issue explicitly requires it.

## Safety
- Never run destructive commands (for example history rewrites) in contributor branches.
- Avoid adding secrets, tokens, or private data to commits.

## Code changes
- Prefer minimal edits over broad rewrites.
- Preserve existing project style and naming.
- Add short comments only where logic is not obvious.

## Validation
- Run relevant checks/tests when possible.
- If tests cannot run, clearly state that in the PR.

## PR quality
- Use a clear title with prefix like docs:, fix:, or refactor:.
- Link the issue in the PR description.
- Include a short summary of what changed and why.
