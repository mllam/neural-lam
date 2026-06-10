# AGENTS.md

Mandatory rules for AI coding agents. Violations will result in rejected PRs.

**Read [CONTRIBUTING.md](CONTRIBUTING.md) first.** It covers the general workflow that applies to
every contributor (open an issue, fork, set up the environment, run `pre-commit` and `pytest`,
fill in the PR template, add a CHANGELOG entry, monthly dev meeting). Everything there applies.
This file adds the AI-specific rules on top.

## Codebase reference

See the README architecture overview for the data flow and module map.
`git log --stat -- neural_lam/` shows which files have moved recently - prefer that over any
snapshot, which will rot.

## AI-specific entry-point commands

The standard install / lint / test commands are in
[CONTRIBUTING.md > Before you push](CONTRIBUTING.md#before-you-push). The two CLI entry points
agents most often need:

```bash
python -m neural_lam.create_graph --config_path <config> --name <graph>
python -m neural_lam.train_model --config_path <config> --model graph_lam --graph <graph>
python -m neural_lam.train_model --eval test --config_path <config> --load <ckpt>
```

W&B is auto-disabled in tests. `DummyDatastore` is the in-memory test fixture; example data is
downloaded from S3 on first run.

---

## AI-specific rules

### Search before creating

Duplicate issues and PRs from AI agents are a recurring problem. Search before opening anything:

```bash
gh issue list --state all --search "<keywords>"
gh pr list --state all --search "<keywords>"
```

If a PR already exists for the same issue, contribute there rather than opening a competing one.

### Re-read the thread before every action

Re-read the full issue / PR thread (including inline review comments via
`gh api repos/mllam/neural-lam/pulls/<N>/comments`) before every comment and every push. Never
repeat a question already answered or an approach already rejected.

### Communication

- **Terse.** One sentence per point. No preamble. No summaries of visible diffs.
- **No filler.** Ban list: "Great question", "As mentioned above", "I hope this helps", "Let me
  know if you have questions", "Happy to help".
- **No obvious narration.** Do not explain what self-explanatory code does.
- **PR descriptions: what changed and why.** Nothing else.
- **One question at a time.** No shotgun lists of open-ended questions.

### Commits

AI attribution is mandatory. Add a `Co-authored-by:` trailer to every commit produced with AI
assistance, e.g.:

Every PR must add a line to `CHANGELOG.md` in the section matching the change type (`Added` / `Changed` / `Fixed` / `Maintenance`).
