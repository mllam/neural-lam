---
name: create-graph
description: Generate and validate graph components and hierarchical meshes using weather-model-graphs. Use when constructing new graph topologies or debugging mesh alignments.
---

# Create and Validate Graph Meshes

Follow these instructions to construct offline graph topologies (Keisler, GraphCast, or Hierarchical) and verify node/edge index alignments.

## 1. Overview & Architecture
Neural-LAM relies on graph representations connecting the regular grid to one or more multiscale mesh levels:
- `g2m`: Grid-to-Mesh bipartite edges.
- `m2g`: Mesh-to-Grid bipartite edges.
- `m2m`: Mesh-to-Mesh intra-level edges.
- `up` / `down`: Inter-level edges for hierarchical multi-scale graphs.

Artifact requirements and disk formats are specified in `docs/graph_storage_spec.md`.

## 2. Programmatic Creation via `neural_lam.api`
Generate graph components in-process using `neural_lam.api.create_graph`:

```python
from neural_lam import api
from tests.dummy_datastore import DummyDatastore

datastore = DummyDatastore(n_grid_points=100, n_timesteps=15)

# Flat single-level mesh (e.g. for GraphLAM)
api.create_graph(
    datastore=datastore,
    name="1level",
    archetype="keisler",
)

# Hierarchical multi-level mesh (e.g. for HiLAM / HiLAMParallel)
api.create_graph(
    datastore=datastore,
    name="hierarchical",
    archetype="hierarchical",
    level_refinement_factor=2,
    mesh_grid_distance_ratio=1.5,
)
```

## 3. CLI Entry Point
Graphs can also be created via CLI:

```bash
python -m neural_lam.create_graph_with_wmg \
  --config_path path/to/config.yaml \
  --name hierarchical \
  --archetype hierarchical
```

## 4. Validate Index and Tensor Alignment
Ensure created graph tensors conform to spec requirements:
- `m2m_edge_index.pt`: Shape `(2, num_m2m_edges)` with indices bounded by `[0, num_mesh_nodes - 1]`.
- `g2m_edge_index.pt`: Shape `(2, num_g2m_edges)` mapping `grid_index -> mesh_index`.
- `m2g_edge_index.pt`: Shape `(2, num_m2g_edges)` mapping `mesh_index -> grid_index`.
- Static features (`m2m_features.pt`, `mesh_features.pt`, etc.) match expected edge counts.

## 5. Verification Check
Run the graph creation test suite:

```bash
uv run pytest tests/test_graph_creation.py
```
