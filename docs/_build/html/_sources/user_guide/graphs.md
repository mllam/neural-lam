# Graph Creation

Neural-LAM uses graph structures to define message-passing GNN layers that
emulate fluid flow in the atmosphere. Graphs are created for a specific datastore
and define the connectivity between grid nodes and mesh nodes.

## Creating Graphs

Use the {func}`~neural_lam.create_graph.cli` command to generate graphs:

```bash
python -m neural_lam.create_graph --config_path <neural-lam-config-path> --name <graph-name>
```

Run `python -m neural_lam.create_graph --help` for a full list of options.

### Graph Types

The different models require different graph structures:

**GC-LAM** (multi-scale mesh):
```bash
python -m neural_lam.create_graph --config_path <config-path> --name multiscale
```

**Hi-LAM / Hi-LAM-Parallel** (hierarchical mesh):
```bash
python -m neural_lam.create_graph --config_path <config-path> --name hierarchical --hierarchical
```

**L1-LAM** (single-level mesh):
```bash
python -m neural_lam.create_graph --config_path <config-path> --name 1level --levels 1
```

## Graph Directory Structure

Generated graphs are stored in the `graphs/` directory:

```
graphs/
├── graph1/
│   ├── m2m_edge_index.pt       - Mesh-to-mesh edges
│   ├── g2m_edge_index.pt       - Grid-to-mesh edges
│   ├── m2g_edge_index.pt       - Mesh-to-grid edges
│   ├── m2m_features.pt         - Mesh edge features
│   ├── g2m_features.pt         - Grid-to-mesh edge features
│   ├── m2g_features.pt         - Mesh-to-grid edge features
│   └── mesh_features.pt        - Mesh node features
├── graph2/
└── ...
```

## Mesh Hierarchy

For hierarchical graphs (`L > 1` levels), files like `m2m_edge_index.pt`,
`m2m_features.pt`, and `mesh_features.pt` contain **lists of length `L`**, where
each entry corresponds to a level (index 0 = lowest level).

Additional files for hierarchical graphs:

```
├── mesh_down_edge_index.pt     - Downward edges between levels
├── mesh_up_edge_index.pt       - Upward edges between levels
├── mesh_down_features.pt       - Downward edge features
├── mesh_up_features.pt         - Upward edge features
```

These list-format files have length `L-1` (connections *between* levels).
Entry 0 describes edges between levels 1 and 2 (the two lowest).
