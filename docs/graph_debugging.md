# Graph Artifacts & Debugging

When you generate a graph using `neural_lam.create_graph`, a set of `.pt` (PyTorch tensor) files is written to the `graphs/<name>` directory. This short guide explains what each file means and how to debug graph-related errors during training.

## Graph Files Overview

The graph structure depends on whether you generated a **1-level/multiscale** graph (flat mesh) or a **hierarchical** graph.

### Core Files (Present in all graphs)
- `g2m_edge_index.pt`: Grid-to-Mesh edges. Encodes data from the input grid onto the processor mesh.
- `m2g_edge_index.pt`: Mesh-to-Grid edges. Decodes processed data from the mesh back to the grid.
- `m2m_edge_index.pt`: Mesh-to-Mesh edges. Internal connections for message passing within the mesh (for a flat graph, this is a list of length 1; for hierarchical, a list of intra-level edges).
- `g2m_features.pt` / `m2g_features.pt` / `m2m_features.pt`: Static edge features (edge length and vector differences) for the respective edges.
- `mesh_features.pt`: Static node features for the mesh (spatial coordinates).

### Hierarchical Files (Only in `hierarchical` graphs)
Hierarchical graphs have multiple mesh levels, requiring vertical message passing:
- `mesh_up_edge_index.pt` / `mesh_down_edge_index.pt`: Edges connecting lower mesh levels to upper mesh levels (up) and vice versa (down).
- `mesh_up_features.pt` / `mesh_down_features.pt`: Static edge features for the vertical edges.

## Edge Features Explained

Edge feature tensors in Neural-LAM contain static geometric information about the connection between two nodes. They typically have a shape of `(num_edges, 3)` where the 3 features are:

1. **`len` (index 0):** The Euclidean distance between the connected nodes, normalized by the longest edge in the graph.
2. **`vdiff_x`, `vdiff_y` (indices 1, 2):** The normalized spatial vector differences (x and y offsets) between the source and destination node.

*(See `create_graph.py` edge generation loops for exact coordinate calculations).*

## Troubleshooting: "Training fails after graph regen"

If your model crashes immediately upon startup or during the first epoch after you regenerated a graph, follow these steps:

1. **Verify File Presence:** Ensure all required `.pt` files (and `metainfo.yaml` for newer specs) exist in your `graphs/<name>` directory.
2. **Check the Specification:** Run the built-in validator script to ensure your graph matches the strict storage specification:
   ```bash
   python docs/validate_graph.py <path_to_datastore> <graph_name>
   ```
   *This tool will automatically verify node counts, edge indices, bounds, and feature shapes (see [graph_storage_spec.md](graph_storage_spec.md)).*
3. **Shape Mismatches:** If the validator passes but training still crashes with dimension mismatches (e.g. `RuntimeError: size mismatch`), double check that the `--config_path` used in `create_graph` matches the datastore used in `train_model`.
