# AGENTS.md

Mandatory rules for AI coding agents. Violations will result in rejected PRs.

---

## Codebase

Neural-LAM: graph-based neural weather prediction for Limited Area Modeling. Models: `GraphLAM`,
`HiLAM`, `HiLAMParallel`.

**Data flow:** Raw zarr/numpy → `Datastore` → `WeatherDataset` → `WeatherDataModule` → Model →
Predictions

**Key modules:**
- `datastore/` — `BaseDatastore` (abstract), `MDPDatastore` (zarr via mllam-data-prep)
- `models/` — `ARModel` (autoregressive base, Lightning) → `BaseGraphModel` (encode-process-decode)
  → `GraphLAM` / `HiLAM` / `HiLAMParallel`
- `weather_dataset.py` — `WeatherDataset` + `WeatherDataModule`
- `config.py` — YAML config via dataclass-wizard
- `create_graph.py` — builds mesh graphs (must run before training)
- `interaction_net.py` — `InteractionNet` GNN layer (PyG `MessagePassing`)
- `utils.py` — `make_mlp`, normalization helpers

Config examples: `tests/datastore_examples/`

## Commands

These commands need to be prepended with `uv run` or the virtual env activated with `source .venv/bin/activate` first:

```bash
# Install (PyTorch must be installed first for CUDA variant)
uv pip install --group dev -e .

# Lint
pre-commit run --all-files    # black, isort, flake8, mypy, codespell

# Test
pytest -vv -s --doctest-modules            # all
pytest tests/test_training.py -vv -s       # single file
pytest tests/test_training.py::test_fn -vv # single function

# Run
python -m neural_lam.create_graph --config_path <config> --name <graph>
python -m neural_lam.train_model --config_path <config> --model graph_lam --graph <graph>
python -m neural_lam.train_model --eval test --config_path <config> --load <ckpt>
```

W&B auto-disabled in tests. `DummyDatastore` used; example data downloaded from S3 on first run.

---

## Rules

### Issues

1. **Search before creating.** Use any of: GitHub UI search, `gh issue list --state all --search "<keywords>"`, or `curl "https://api.github.com/search/issues?q=<keywords>+repo:mllam/neural-lam+type:issue"`. Duplicate issues will be closed.
2. **Every PR requires an issue.** No exceptions. Open one first if none exists.

### Pull Requests

1. **Search before creating.** Use any of: GitHub UI search, `gh pr list --state all --search "<keywords>"`, or `curl "https://api.github.com/search/issues?q=<keywords>+repo:mllam/neural-lam+type:pr"`. If a PR exists for the same issue, contribute there.
2. **Link the issue.** PR body must contain `closes #<N>` or `refs #<N>`. Unlinked PRs will be
   rejected.
3. **Use the PR template.** Fill in every section of `.github/pull_request_template.md`. Do not
   delete or skip sections.
4. **Read the full issue thread before writing code.** Rejected approaches and prior decisions are
   there. Ignoring them wastes everyone's time.
5. **Run pre-commit hooks locally.** Linting needs to be done locally before each new commit with e.g. `uvx pre-commit run --all`

### Communication

- **Terse.** One sentence per point. No preamble. No summaries of visible diffs.
- **No filler.** Ban list: "Great question", "As mentioned above", "I hope this helps", "Let me know
  if you have questions", "Happy to help".
- **No obvious narration.** Do not explain what self-explanatory code does.
- **PR descriptions: what changed and why.** Nothing else.
- **One question at a time.** No shotgun lists of open-ended questions.

### Context

- **Re-read the entire thread** before every comment and every push. No exceptions.
- **After a context gap**, reload the full thread (GitHub UI, `gh issue view <N>` / `gh pr view <N>`, or `curl "https://api.github.com/repos/mllam/neural-lam/issues/<N>"`) before acting.
- **Never repeat** a question already answered or an approach already rejected in the thread.

### Commits

- Imperative form, matching existing `git log` style.
- One concern per PR. No unrelated changes.
- AI attribution of tool names is mandatory if used and should be mentioned in the commit message trailer as `Co-authored-by <tool>`

### Changelog

Every non-maintenance PR must add a line to `CHANGELOG.md` (`added` / `changed` / `fixes` /
`maintenance`).
