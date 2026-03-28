# AGENTS.md

Mandatory rules and context for AI coding agents. 
**Human contributors: Please refer to the [Contributing Guide](docs/contributing/contributing.md).**

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

## Mandate for AI Agents

1. **Follow Contributing Guide.** AI agents must adhere to the rules in [CONTRIBUTING.md](docs/contributing/contributing.md) (Issues, PRs, Communication style).
2. **AI Attribution.** Mandatory `Co-authored-by <tool>` in commit trailers when AI tools are used.
3. **Context Gap.** Reload the full thread (GitHub UI or `gh` CLI) after a context gap before acting.
4. **No Placeholders.** Do not use placeholders in code or documentation.
