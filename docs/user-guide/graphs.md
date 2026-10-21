# Graph Creation

## Graph creation with weather-model-graphs (recommended)

The recommended way to create graphs is with the `create_graph_with_wmg`
command, which delegates graph construction to
[weather-model-graphs](https://github.com/mllam/weather-model-graphs):

```bash
python -m neural_lam.create_graph_with_wmg --config_path <neural-lam-config-path> --archetype <archetype> --name <graph-name>
```

Available archetypes:

| Archetype | Model | Description |
|---|---|---|
| `keisler` | `graph_lam` | Single-level mesh (default) |
| `graphcast` | `graph_lam` | Multi-scale mesh |
| `hierarchical` | `hi_lam` / `hi_lam_parallel` | Hierarchical mesh |

Run `python -m neural_lam.create_graph_with_wmg --help` for the full list of
options (e.g. `--mesh_node_distance`, `--mesh_grid_distance_ratio`,
`--level_refinement_factor`, `--max_num_levels`).

## Graph creation with `create_graph` (deprecated)

> **Note:** The `create_graph` command below is deprecated and will be removed
> in a future release. Please use `create_graph_with_wmg` (see above) instead.

Run `python -m neural_lam.create_graph` with suitable options to generate the graph you want to use (see `python -m neural_lam.create_graph --help` for a list of options).
The graphs used for the different models in the [paper](https://arxiv.org/abs/2309.17370) can be created as:

* **GC-LAM**: `python -m neural_lam.create_graph --config_path <neural-lam-config-path> --name multiscale`
* **Hi-LAM**: `python -m neural_lam.create_graph --config_path <neural-lam-config-path> --name hierarchical --hierarchical` (also works for Hi-LAM-Parallel)
* **L1-LAM**: `python -m neural_lam.create_graph --config_path <neural-lam-config-path> --name 1level --levels 1`

The graph-related files are stored in a directory called `graph`.

## Graph storage format

For a strict, machine-verifiable specification of the graph storage format, see the {doc}`../graph_storage_spec`.
