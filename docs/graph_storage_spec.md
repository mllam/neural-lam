# Neural-LAM Graph Storage Specification

Version: 0.1.0-draft

## 1. Introduction

This document specifies the requirements for Graph disk format for `neural-lam`.
These graphs are used by the Neural-LAM Graph Neural Network architectures for
machine-learning weather prediction (MLWP) forecasting. These model
architectures follow the encode-process-decode paradigm of sequential message
passing, where physical variables are represented as features on so-called
*grid* nodes, are *encoded* to *mesh* nodes, are *processed* on the mesh, and
then *decoded* back to grid nodes where output tendencies or updated state are
produced.

The format specified in this document was designed to support the definition of
both flat (e.g. Keisler 2022, Lam et al 2022) and hierarchical (Oskarsson et al
2023) graphs for GNN-based MLWP in neural-lam.

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT",
"SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this
document are to be interpreted as described in RFC 2119.

## 2. File and Directory Structure

### Directory Structure

Each graph MUST identified by a unique `name` and stored within the directory
`graph/<name>/` that in turn MUST be placed within the same directory as the
datastore configuration from which the graph was derived (i.e. the spatial
coordinates defining the `grid` coordinates are provided by the datastore).

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

Every tensors MUST stored in a manner ameanable to load with `torch.load(...)` (this can most easily be support by using `torch.save(...)` to store tensors to disk) and satisfy the requirements below.

### Nodes

The `neural-lam` graph format on disk does not explicitly store node features for grid nodes, as these are expected to be dynamic and stored separately in the dataset. However, static features for mesh nodes MUST be stored in `mesh_features.pt` files (as described below).

#### Node index space

Node indices in edge tensors MUST be stored in a single global index space:

- Mesh nodes MUST come first.
- Grid nodes MUST follow after all mesh nodes.
- For hierarchical graphs, each mesh level MUST occupy a contiguous mesh index
  range in ascending level order.

#### Mesh node features

`mesh_features.pt` files MUST be lists of length `L` (number of mesh levels),
where each entry is a tensor containing static features for the mesh nodes at
that level. Each tensor MUST satisfy the following requirements:

- `mesh_features` entries MUST have shape `[N_level, 2]`, where `N_level` is the number of mesh nodes at that level.
- Columns MUST be x/y coordinates.
- In graphs created by `neural_lam.create_graph`, coordinates are normalized by
  dividing by the maximum absolute grid coordinate.
- Dtype MUST be `torch.float32`.

### Edges

#### Edge indices

The following edge index files MUST be defined:

- `g2m_edge_index.pt`
- `m2g_edge_index.pt`
- `m2m_edge_index.pt`
- `mesh_up_edge_index.pt` (hierarchical graphs only, `L > 1`)
- `mesh_down_edge_index.pt` (hierarchical graphs only, `L > 1`)

`g2m_edge_index.pt` and `m2g_edge_index.pt` MUST each contain a single tensor
with shape `[2, E]`, where `E` is the number of edges in that component.

`m2m_edge_index.pt` MUST contain a list of tensors of length `L`, i.e. one
edge-index tensor per mesh level. Each entry MUST have shape `[2, E_level]`,
where `E_level` is the number of edges at that level.

For hierarchical graphs, `mesh_up_edge_index.pt` and
`mesh_down_edge_index.pt` MUST each contain a list of length `L - 1` of
tensors, i.e. one per inter-level connection, so that entry `i` connects level
`i` and level `i+1`. Each entry MUST have shape `[2, E_interlevel]`, where
`E_interlevel` is the number of edges going either up
(`mesh_up_edge_index.pt`) or down (`mesh_down_edge_index.pt`) between that
level pair.

For every edge-index tensor above:

- Row `0` MUST be sender node index, row `1` MUST be receiver node index.
- Dtype MUST be `torch.int64`.

#### Edge features

The following edge feature files MUST be defined:

- `g2m_features.pt`
- `m2g_features.pt`
- `m2m_features.pt`
- `mesh_up_features.pt` (hierarchical graphs only, `L > 1`)
- `mesh_down_features.pt` (hierarchical graphs only, `L > 1`)

`g2m_features.pt` and `m2g_features.pt` MUST each contain a single tensor with
shape `[E, N_f]`, where `E` matches the number of edges in the corresponding
`*_edge_index.pt` file.

`m2m_features.pt` MUST contain a list of length `L`, i.e. one feature tensor
per mesh level. Entry `i` MUST have shape `[E_level, N_f]`, where `E_level`
matches the edge count in entry `i` of `m2m_edge_index.pt`.

For hierarchical graphs, `mesh_up_features.pt` and `mesh_down_features.pt`
MUST each contain a list of length `L - 1`, i.e. one feature tensor per
inter-level connection between level `i` and `i+1`. Entry `i` MUST have shape
`[E_interlevel, N_f]`, where `E_interlevel` matches the edge count in entry `i`
of the corresponding `mesh_*_edge_index.pt` file.

For every edge feature tensor above:

- The shape MUST be `[E_component, N_f]`.
- `N_f` MUST be consistent across all edge feature tensors in the graph.
- Dtype MUST be `torch.float32`.

In graphs created by `neural_lam.create_graph`, `N_f == 3` by default with:

- Column `0`: edge length, where `edge_length = ||sender_pos - receiver_pos||_2`.
- Column `1`: x-component of `vdiff = sender_pos - receiver_pos`.
- Column `2`: y-component of `vdiff = sender_pos - receiver_pos`.
