# Contributing Guidelines

Mandatory rules for contributing to Neural-LAM. Follow these guidelines to ensure your Pull Requests are accepted.

---

## Issues

1. **Search before creating.** Use any of: GitHub UI search, `gh issue list --state all --search "<keywords>"`, or `curl "https://api.github.com/search/issues?q=<keywords>+repo:mllam/neural-lam+type:issue"`. Duplicate issues will be closed.
2. **Every PR requires an issue.** No exceptions. Open one first if none exists.
3. **Include minimal example.** Each issue should include a minimal, reproducible example on how to easily recreate a bug, including all necessary module imports and data. Include full traceback if it is a bug-report.

## Pull Requests

1. **Search before creating.** Use any of: GitHub UI search, `gh pr list --state all --search "<keywords>"`, or `curl "https://api.github.com/search/issues?q=<keywords>+repo:mllam/neural-lam+type:pr"`. If a PR exists for the same issue, contribute there.
2. **Link the issue.** PR body must contain `closes #<N>` or `refs #<N>`. Unlinked PRs will be
   rejected.
3. **Use the PR template.** Fill in every section of `.github/pull_request_template.md`. Do not
   delete or skip sections.
4. **Read the full issue thread before writing code.** Rejected approaches and prior decisions are
   there. Ignoring them wastes everyone's time.
5. **Run pre-commit hooks locally.** Linting needs to be done locally before each new commit with e.g. `uvx pre-commit run --all`
6. **Testing Mandate.** Run `pytests tests/` before opening a PR and if tests fail do not open the PR , fix the failure first.

## Communication

- **Terse.** One sentence per point. No preamble. No summaries of visible diffs.
- **No filler.** Ban list: "Great question", "As mentioned above", "I hope this helps", "Let me know
  if you have questions", "Happy to help".
- **No obvious narration.** Do not explain what self-explanatory code does.
- **PR descriptions: what changed and why.** Nothing else.
- **One question at a time.** No shotgun lists of open-ended questions.

## Context

- **Re-read the entire thread** before every comment and every push. No exceptions.
- **After a context gap**, reload the full thread (GitHub UI, `gh issue view <N>` / `gh pr view <N>`, or `curl "https://api.github.com/repos/mllam/neural-lam/issues/<N>"`) before acting.
- **Never repeat** a question already answered or an approach already rejected in the thread.

## Commits & Changelog

- **Commit Format.** Imperative form, matching existing `git log` style.
- **One Concern per PR.** No unrelated changes.
- **AI Attribution.** Mandatory `Co-authored-by <tool>` in commit trailers if AI tools are used.
- **CHANGELOG.** Every PR must add a line to `CHANGELOG.md` in the section matching the change type (`Added` / `Changed` / `Fixed` / `Maintenance`).

---
*For a technical overview intended for AI agents, see [AGENTS.md](AGENTS.md).*
