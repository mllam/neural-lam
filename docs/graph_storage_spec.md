# Neural-LAM Graph Storage Specification

This document specifies the on-disk graph format expected by
`neural_lam.utils.load_graph` and the graph-based models.

## File and Directory Structure

### Directory Structure

Each graph lives in a directory (typically `<datastore-root>/graph/<name>`).

### Graph Filenames

Each graph is represented by multiple files in its graph directory.
Together, these files define connectivity and static features for edges and
mesh nodes.

Required files for all graphs:

- `m2m_edge_index.pt`
- `g2m_edge_index.pt`
- `m2g_edge_index.pt`
- `m2m_features.pt`
- `g2m_features.pt`
- `m2g_features.pt`
- `mesh_features.pt`

Additional required files for hierarchical graphs (`L > 1` mesh levels):

- `mesh_up_edge_index.pt`
- `mesh_down_edge_index.pt`
- `mesh_up_features.pt`
- `mesh_down_features.pt`

The separate files represent the different "components" of the graph, where each of the sequential message passing steps, `encode`, `process`, and `decode` uses a separate component, so that `g2m` is used in `encode`, `m2m` is used in `process`, and `m2g` is used in `decode`. For hierarchical graphs, the `m2m` component is further split into separate inter-level and intra-level message-passing steps.

Each graph component is represented by two files: one for edge connectivity and one for edge features. The components are (which also define the expected file prefixes):

- `g2m`: grid-to-mesh edges (sender on grid, receiver on mesh).
- `m2m`: mesh-to-mesh edges (both sender and receiver on mesh).
- `m2g`: mesh-to-grid edges (sender on mesh, receiver on grid).
- `mesh_up`: inter-level mesh edges from lower level to upper level.
- `mesh_down`: inter-level mesh edges from upper level to lower level.

Suffixes indicate content type:

- `_edge_index.pt`: edge connectivity
- `_features.pt`: static features associated with each edge
- `mesh_features.pt`: static mesh node features

All files are serialized with `torch.save(...)`.

> **NOTE**: Rather than the inter-level tensors files being prefixed with `m2m`, they are prefixed with `mesh`, even though they are part of the mesh-to-mesh message passing.

## Tensor content requirements

### Edges

#### Edge indices

`*_edge_index.pt` files are each a single tensor containing edge connectivity information for a particular graph component. The tensor must satisfy the following requirements:

- The shape must be `[2, E]`, where `E` is the number of edges in that component.
- Row `0` is sender node index, row `1` is receiver node index.
- Dtype must be integer.

#### Edge features

`*_features.pt` files are each a single tensor containing static features for the edges in a particular graph component. The tensor must satisfy the following requirements:

- The shape must be `[E, 3]`, where `E` is the number of edges in that component (must match the corresponding edge index file).
- Columns `1:3`: `vdiff = sender_pos - receiver_pos` in x/y.
- Column `0`: edge length, where `edge_length = ||vdiff||_2`.
- Dtype floating-point.

### Nodes

#### Mesh node features

`mesh_features.pt` files must be lists of length `L` (number of mesh levels), where each entry is a tensor containing static features for the mesh nodes at that level. Each tensor must satisfy the following requirements:

- `mesh_features` entries have shape `[N_level, 2]`, dtype floating-point.
- Columns are x/y coordinates.
- In graphs created by `neural_lam.create_graph`, coordinates are normalized by
  dividing by the maximum absolute grid coordinate.

## Level Encoding

`m2m_edge_index.pt`, `m2m_features.pt`, and `mesh_features.pt` store lists of
length `L` (number of mesh levels).

- Non-hierarchical graphs: `L == 1`.
- Hierarchical graphs: `L > 1`.
- Entry `0` is always the bottom mesh level.

For hierarchical graphs, `mesh_up_*` and `mesh_down_*` are lists of length
`L - 1`, where entry `i` connects level `i` and level `i+1`.

## Node Index Space

Node indices in edge tensors are stored in a single global index space:

- Mesh nodes come first.
- Grid nodes follow after all mesh nodes.
- For hierarchical graphs, each mesh level occupies a contiguous mesh index
  range in ascending level order.

Additional conventions used by `neural_lam.create_graph`:

- `g2m` receivers are on the bottom mesh level.
- `m2g` senders are on the bottom mesh level.

## Runtime Load Expectations

`neural_lam.utils.load_graph` applies these assumptions:

- It always expects the files listed above.
- It infers `hierarchical = (len(m2m_edge_index) > 1)`.
- For non-hierarchical graphs (`L == 1`), it unwraps single-entry lists into
  plain tensors for `m2m_*` and `mesh_features`.
- For hierarchical graphs (`L > 1`), it keeps per-level tensors in
  `BufferList` containers.
- It normalizes all edge features by the longest edge length found in
  `m2m_features`.

## Validation CLI

Use:

```bash
uv run docs/validate_graph.py --graph_dir <path-to-graph-dir>
```

The validator loads each component from disk and checks:

- required files and container types,
- shape/dtype consistency,
- cross-file edge-count consistency,
- mesh-level consistency,
- hierarchical up/down level connectivity ranges,
- grid/mesh index partitioning conventions,
- consistency with the runtime graph tensor conventions.
