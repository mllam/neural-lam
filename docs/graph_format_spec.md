# Neural-LAM Graph Format Specification

This document describes the expected on-disk graph format used by **neural-lam**.
The graph represents the encode–process–decode architecture used by the model.

The graph is stored as serialized PyTorch tensors (`.pt` files) inside:

```
<datastore_root>/graph/<graph_name>/
```

---

# Graph Structure Overview

The neural-lam architecture operates on a graph composed of two node types:

**Grid nodes**

* Represent the spatial weather grid.
* Contain atmospheric state variables.

**Mesh nodes**

* Latent processing nodes used by the neural network.
* Enable multi-scale message passing.

The graph contains three primary edge types:

```
Grid → Mesh (g2m)
Mesh → Mesh (m2m)
Mesh → Grid (m2g)
```

These correspond to the **encode–process–decode** architecture.

---

# Required Files

For all graphs the following files must exist:

```
g2m_edge_index.pt
g2m_features.pt

m2g_edge_index.pt
m2g_features.pt

m2m_edge_index.pt
m2m_features.pt

mesh_features.pt
```

---

# Edge Index Format

Edge indices follow the PyTorch Geometric convention:

```
edge_index.shape = (2, N_edges)
```

Where:

```
edge_index[0] = sender nodes
edge_index[1] = receiver nodes
```

---

# Edge Feature Format

Edge feature tensors have shape:

```
(N_edges, d_edge_features)
```

Currently the expected feature layout is:

```
[edge_length, delta_x, delta_y]
```

Where:

| Column | Description             |
| ------ | ----------------------- |
| 0      | Euclidean edge length   |
| 1      | x-coordinate difference |
| 2      | y-coordinate difference |

The loader assumes **column 0 contains the physical edge length**.

---

# Mesh Node Features

```
mesh_features.pt
```

Contains the static mesh node features.

Shape:

```
(N_mesh_nodes, d_mesh_features)
```

Currently:

```
[x, y]
```

representing spatial coordinates.

---

# Hierarchical Graphs

If hierarchical graphs are used, additional files must exist:

```
mesh_up_edge_index.pt
mesh_up_features.pt

mesh_down_edge_index.pt
mesh_down_features.pt
```

These define connections between mesh levels.

Each is stored as a **list of tensors**, one per hierarchy level.

---

# Node Ordering

The graph assumes the following ordering of node indices:

```
[mesh nodes]
[grid nodes]
```

Mesh nodes must occupy the **lowest index range**.

Example:

```
mesh nodes : 0 ... N_mesh-1
grid nodes : N_mesh ... N_total-1
```

This ordering is relied upon by the message passing implementation.

---

# Normalization

Edge features are normalized during graph loading by dividing by the **longest edge length** in the mesh-to-mesh graph.

This ensures edge feature values remain within a stable numeric range for training.

---

# Purpose

This specification exists to:

* make the graph format explicit
* simplify interoperability with external graph generators
* prevent silent errors when generating graphs

External tools generating graphs for neural-lam should follow this specification.
