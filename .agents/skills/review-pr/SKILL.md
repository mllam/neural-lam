---
name: review-pr
description: Review a Neural-LAM Pull Request following maintainer checklists, comment endpoints, and label rules.
---

# Review a Pull Request

Follow these guidelines when reviewing a Pull Request in `neural-lam`.

## 1. Inspect Complete Review Thread
Always fetch both top-level discussion and inline code review comments per `AGENTS.md#re-read-the-thread-before-every-action`:
```bash
# Top-level thread and PR description
gh pr view <PR_NUMBER> --comments

# Inline code review comments
gh api repos/mllam/neural-lam/pulls/<PR_NUMBER>/comments
```
Never repeat a question already answered or an approach already rejected in the thread.

## 2. Review Checklists
Follow the checklists defined in `CONTRIBUTING.md#pull-requests` and `.github/pull_request_template.md`:
- **Automated pre-flight**: Verify CI status (`gh pr checks <PR_NUMBER>`), `CHANGELOG.md` entry, test coverage, and mandatory AI attribution trailers (`AGENTS.md#commits`).
- **Maintainer judgment**: Verify architectural alignment, ensure changes stay within scope without unrequested refactorings, and assign appropriate labels and milestones per `CONTRIBUTING.md#community-roadmap`.

## 3. Communication & Feedback
Adhere to the communication standards in `AGENTS.md#communication`:
- Keep feedback terse: one sentence per point, no preamble, no filler.
- Use GitHub markdown suggestion blocks for concrete code adjustments:
  ````markdown
  ```suggestion
      self.loss = metrics.get_metric(loss)
  ```
  ````
