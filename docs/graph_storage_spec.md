# Neural-LAM Graph Storage Specification

Version: 0.1.0

## 1. Introduction

This document specifies the requirements for Graph disk format for neural-lam.
The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT",
"SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this
document are to be interpreted as described in RFC 2119.

## 2. File and Directory Structure

### Directory Structure

Each graph MUST live in a directory (typically
`<datastore-root>/graph/<name>`).

### Graph Filenames

Each graph MUST be represented by multiple files in its graph directory.
Together, these files define connectivity and static features for edges and
mesh nodes.

Required files for all graphs (all of these files MUST be present):

- `m2m_edge_index.pt`
- `g2m_edge_index.pt`
- `m2g_edge_index.pt`
- `m2m_features.pt`
- `g2m_features.pt`
- `m2g_features.pt`
- `mesh_features.pt`

Additional required files for hierarchical graphs (`L > 1` mesh levels), all
of which MUST be present:

- `mesh_up_edge_index.pt`
- `mesh_down_edge_index.pt`
- `mesh_up_features.pt`
- `mesh_down_features.pt`

The separate files represent the different "components" of the graph, where
each of the sequential message passing steps, `encode`, `process`, and
`decode` uses a separate component, so that `g2m` is used in `encode`, `m2m`
is used in `process`, and `m2g` is used in `decode`. For hierarchical graphs,
the `m2m` component is further split into separate inter-level and intra-level
message-passing steps.

Each graph component MUST be represented by two files: one for edge
connectivity and one for edge features. The components are (which also define
the expected file prefixes):

- `g2m`: grid-to-mesh edges (sender on grid, receiver on mesh).
- `m2m`: mesh-to-mesh edges (both sender and receiver on mesh).
- `m2g`: mesh-to-grid edges (sender on mesh, receiver on grid).
- `mesh_up`: inter-level mesh edges from lower level to upper level.
- `mesh_down`: inter-level mesh edges from upper level to lower level.

Suffixes indicate content type:

- `_edge_index.pt`: edge connectivity
- `_features.pt`: static features associated with each edge
- `mesh_features.pt`: static mesh node features

All files MUST be serialized with `torch.save(...)`.

> **NOTE**: Rather than the inter-level tensors files being prefixed with
> `m2m`, they are prefixed with `mesh`, even though they are part of the
> mesh-to-mesh message passing.

## 3. File content requirements

The content of the files depend on the number of mesh levels, denoted as `L` in
the text below, so that for:

- Non-hierarchical graphs `L == 1`.
- Hierarchical graphs `L > 1`.
- Entry `0` MUST always be the bottom mesh level.

### Edges

#### Edge indices

`g2m_edge_index.pt` and `m2g_edge_index.pt` MUST each contain a single tensor
with shape `[2, E]`, where `E` is the number of edges in that component.

`m2m_edge_index.pt` MUST contain a list of tensors of length `L`, i.e. one
edge-index tensor per mesh level. Each entry MUST have shape `[2, E_level]`
where `E_level` is the number of edges at that level.

For hierarchical graphs, `mesh_up_edge_index.pt` and
`mesh_down_edge_index.pt` MUST each contain a list of length `L - 1` of tensors,
i.e. one per inter-level connection, so that  entry `i` connects level `i` and
level `i+1`. Each entry MUST have shape `[2, E_interlevel]` where `E_interlevel` is the number of edges going either up
(`mesh_up_edge_index.pt`) or down (`mesh_down_edge_index.pt`) between that
level pair.

For every edge-index tensor above:

- Row `0` MUST be sender node index, row `1` MUST be receiver node index.
- Dtype MUST be integer.

#### Edge features

`*_features.pt` files MUST each contain a single tensor with static features
for the edges in a particular graph component. The tensor MUST satisfy the
following requirements:

- The shape MUST be `[E, 3]`, where `E` is the number of edges in that
  component (and MUST match the corresponding edge index file).
- The graph generation in [neural-lam v0.1.0]() creates `E==3` edge features with the following content:
  - Columns `1:3` MUST be `vdiff = sender_pos - receiver_pos` in x/y.
  - Column `0` MUST be edge length, where `edge_length = ||vdiff||_2`.
- Dtype MUST be floating-point.

### Nodes

#### Mesh node features

`mesh_features.pt` files MUST be lists of length `L` (number of mesh levels),
where each entry is a tensor containing static features for the mesh nodes at
that level. Each tensor MUST satisfy the following requirements:

- `mesh_features` entries MUST have shape `[N_level, 2]`, where `N_level` is the number of mesh nodes at that level.
- Columns MUST be x/y coordinates.
- In graphs created by `neural_lam.create_graph`, coordinates are normalized by
  dividing by the maximum absolute grid coordinate.
- Dtype MUST be floating-point.

## 4. Level Encoding

`m2m_edge_index.pt`, `m2m_features.pt`, and `mesh_features.pt` MUST store lists of
length `L` (number of mesh levels).

- Non-hierarchical graphs: `L == 1`.
- Hierarchical graphs: `L > 1`.
- Entry `0` MUST always be the bottom mesh level.


## 5. Node Index Space

Node indices in edge tensors MUST be stored in a single global index space:

- Mesh nodes MUST come first.
- Grid nodes MUST follow after all mesh nodes.
- For hierarchical graphs, each mesh level MUST occupy a contiguous mesh index
  range in ascending level order.

Additional conventions used by `neural_lam.create_graph`:

- `g2m` receivers MUST be on the bottom mesh level.
- `m2g` senders MUST be on the bottom mesh level.

## 6. Runtime Load Expectations

`neural_lam.utils.load_graph` applies these assumptions (graph files MUST
conform):

- It expects the files listed above.
- It infers `hierarchical = (len(m2m_edge_index) > 1)`.
- For non-hierarchical graphs (`L == 1`), it unwraps single-entry lists into
  plain tensors for `m2m_*` and `mesh_features`.
- For hierarchical graphs (`L > 1`), it keeps per-level tensors in
  `BufferList` containers.
- It normalizes all edge features by the longest edge length found in
  `m2m_features`.

## 7. Validation CLI

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
