# Graph Creation

Neural-LAM uses graph structures to define the message-passing layers of its
GNN models. A graph must be created for the specific datastore and model
architecture you intend to use.

## Creating graphs with `create_graph_with_wmg` (recommended)

The recommended way to create graphs delegates construction to
[weather-model-graphs](https://github.com/mllam/weather-model-graphs):

```bash
python -m neural_lam.create_graph_with_wmg \
    --config_path <neural-lam-config-path> \
    --archetype <archetype> \
    --name <graph-name>
```

Available archetypes:

| Archetype | Model | Description |
|---|---|---|
| `keisler` | `graph_lam` | Single-level mesh (default) |
| `graphcast` | `graph_lam` | Multi-scale mesh |
| `hierarchical` | `hi_lam` / `hi_lam_parallel` | Hierarchical mesh |

Additional options (see `--help` for the full list):

- `--mesh_node_distance` — controls mesh node density
- `--mesh_grid_distance_ratio` — ratio of mesh node distance to grid spacing
- `--level_refinement_factor` — refinement factor between hierarchy levels
- `--max_num_levels` — maximum number of hierarchy levels

Graph files are saved under `graphs/<graph-name>/` relative to the config
root directory.

## Legacy graph creation with `create_graph`

```{deprecated}
`create_graph` is deprecated and will be removed in a future release.
Use `create_graph_with_wmg` instead.
```

The original `create_graph` command is still available:

```bash
# GC-LAM
python -m neural_lam.create_graph \
    --config_path <config> --name multiscale

# Hi-LAM / Hi-LAM-Parallel
python -m neural_lam.create_graph \
    --config_path <config> --name hierarchical --hierarchical

# L1-LAM
python -m neural_lam.create_graph \
    --config_path <config> --name 1level --levels 1
```

## Graph storage format

Generated graphs are stored under `graphs/<graph-name>/`:

```
graphs/
└── keisler/
    ├── m2m_edge_index.pt     - Mesh-to-mesh edges
    ├── g2m_edge_index.pt     - Grid-to-mesh edges
    ├── m2g_edge_index.pt     - Mesh-to-grid edges
    ├── m2m_features.pt       - Static mesh edge features
    ├── g2m_features.pt       - Static grid-to-mesh edge features
    ├── m2g_features.pt       - Static mesh-to-grid edge features
    └── mesh_features.pt      - Static mesh node features
```

For hierarchical graphs (`L > 1` levels), additional files are included:

```
    ├── mesh_down_edge_index.pt   - Downward inter-level edges
    ├── mesh_up_edge_index.pt     - Upward inter-level edges
    ├── mesh_down_features.pt     - Static downward edge features
    └── mesh_up_features.pt       - Static upward edge features
```

The files `m2m_edge_index.pt`, `m2m_features.pt`, and `mesh_features.pt`
contain lists of length `L` (one entry per level). For non-hierarchical graphs
`L == 1`. Level 0 (index 0) is the lowest level in the hierarchy.
The `mesh_down_*` and `mesh_up_*` files have length `L-1`.

For the full machine-verifiable specification see {doc}`../graph_storage_spec`.
