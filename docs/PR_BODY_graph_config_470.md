## Summary

- Write `graph_config.json` at the end of `create_graph` with generation parameters, mesh-tree summary (`branching_children_per_axis`, `nlev_full_tree`, `nleaf`), g2m `dm_scale`, tensor layout dims, `neural_lam_version`, UTC `generated_at_utc`, and optional `git_commit` (from `GITHUB_SHA` / `git rev-parse`).
- `validate_graph_dir` / `validate_graph` CLI: if `graph_config.json` exists, validate JSON + required keys + schema version; `--require_graph_config` fails when the file is missing.

Closes #470

## How to test

```bash
pytest tests/test_graph_config.py tests/test_validate_graph.py -q
```
