# AGENTS.md

Mandatory rules for AI coding agents. Violations will result in rejected PRs.

**Read [CONTRIBUTING.md](CONTRIBUTING.md) first.** It covers the general workflow that applies to
every contributor (open an issue, fork, set up the environment, run `pre-commit` and `pytest`,
fill in the PR template, add a CHANGELOG entry, monthly dev meeting). Everything there applies.
This file adds the AI-specific rules on top.

---

## Codebase

Neural-LAM: graph-based neural weather prediction for Limited Area Modeling. Models: `GraphLAM`,
`HiLAM`, `HiLAMParallel`.

**Data flow:** Raw zarr/numpy → `Datastore` → `WeatherDataset` → `WeatherDataModule` → Model →
Predictions

**Key modules:**
- `datastore/` — `BaseDatastore` (abstract), `MDPDatastore` (zarr via mllam-data-prep)
- `models/` — `ForecasterModule` (Lightning) → `Forecaster` (`ARForecaster`) →
  `StepPredictor` (`GraphLAM` / `HiLAM` / `HiLAMParallel`)
- `weather_dataset.py` — `WeatherDataset` + `WeatherDataModule`
- `config.py` — YAML config via dataclass-wizard
- `create_graph.py` — builds mesh graphs (must run before training)
- `interaction_net.py` — `InteractionNet` GNN layer (PyG `MessagePassing`)
- `utils.py` — `make_mlp`, normalization helpers

Config examples: `tests/datastore_examples/`

## Commands

Prepend `uv run` or activate the venv first with `source .venv/bin/activate`:

```bash
# Install (PyTorch must be installed first for CUDA variant)
uv pip install --group dev -e .

# Lint
pre-commit run --all-files    # black, isort, flake8, mypy, codespell

# Test
pytest -vv -s --doctest-modules            # all
pytest tests/test_training.py -vv -s       # single file
pytest tests/test_training.py::test_fn -vv # single function
pytest -m "not slow"                       # skip long-running training tests

# Run
python -m neural_lam.create_graph --config_path <config> --name <graph>
python -m neural_lam.train_model --config_path <config> --model graph_lam --graph <graph>
python -m neural_lam.train_model --eval test --config_path <config> --load <ckpt>
```

W&B auto-disabled in tests. `DummyDatastore` used; example data downloaded from S3 on first run.

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

- Re-read the entire issue / PR thread before every comment and every push. No exceptions.
- After a context gap, reload it (`gh issue view <N>` / `gh pr view <N>` /
  `gh api repos/mllam/neural-lam/pulls/<N>/comments`) before acting.
- Never repeat a question already answered or an approach already rejected in the thread.

### Communication

- **Terse.** One sentence per point. No preamble. No summaries of visible diffs.
- **No filler.** Ban list: "Great question", "As mentioned above", "I hope this helps", "Let me
  know if you have questions", "Happy to help".
- **No obvious narration.** Do not explain what self-explanatory code does.
- **PR descriptions: what changed and why.** Nothing else.
- **One question at a time.** No shotgun lists of open-ended questions.

### Commits

- AI attribution is mandatory. Add a `Co-authored-by: <tool> <noreply@...>` trailer to every
  commit produced with AI assistance.
