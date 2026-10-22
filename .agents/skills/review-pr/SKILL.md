---
name: review-pr
description: Review a Neural-LAM Pull Request following maintainer checklists, comment endpoints, and label rules. Use when reviewing incoming contributions.
---

# Review a Pull Request

Follow these instructions when reviewing a Pull Request in `neural-lam`.

## 1. Fetch Complete Review Thread
Always inspect both inline review comments and top-level thread discussions before taking action:

```bash
# View top-level conversation and PR description
gh pr view <PR_NUMBER> --comments

# Fetch inline code review comments
gh api repos/mllam/neural-lam/pulls/<PR_NUMBER>/comments
```

Never repeat a question already answered or an approach already rejected in the thread.

## 2. Review Checklist: Automated vs. Maintainer Judgment
Distinguish between automated checks and manual maintainer evaluations:

### Automated Pre-Flight Checks
Verify objective criteria using CLI tooling:
1. **CI Status**: Verify all CI jobs (lint, unit tests, docs) pass:
   ```bash
   gh pr checks <PR_NUMBER>
   ```
2. **Changelog**: Verify that `CHANGELOG.md` contains an entry in the correct section (`Added`, `Changed`, `Fixed`, or `Maintenance`) with the PR link.
3. **Tests**: Ensure tests are present for newly introduced models, datastores, metrics, or graph features.
4. **AI Attribution**: If AI tooling was used, verify commits include `Co-authored-by:` trailers.

### Maintainer Judgment Responsibilities
Assess subjective and architectural requirements:
1. **Scope & Design**: Confirm the PR addresses the issue without introducing unrequested architecture or scope creep.
2. **Discussion Context**: Ensure the approach aligns with Dev Meeting agreements and previous thread decisions.
3. **Labels & Milestones**: Assign the appropriate labels and milestones via GitHub CLI:
   ```bash
   gh pr edit <PR_NUMBER> --add-label "enhancement"
   ```

## 3. Formatting Suggestions and Feedback
- Keep comments terse: one sentence per point, no filler, no visible diff summaries.
- Use GitHub markdown suggestion blocks for concrete code changes:

````markdown
```suggestion
    self.loss = metrics.get_metric(loss)
```
````
