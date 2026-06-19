# Graph Construction

## Why Graphs?

Graph Neural Networks (GNNs) leverage graph-based message passing, which is highly suited for weather prediction. Unlike standard CNNs on rigid grids, graphs can naturally represent irregular spatial distributions and complex geometries, common in Limited Area Modeling (LAM).

The graph structure dictates how information flows across spatial regions. A well-designed mesh ensures that localized weather phenomena correctly influence neighboring regions, while hierarchical structures allow long-range interactions (like large-scale pressure systems) to propagate efficiently across the domain without requiring hundreds of standard grid steps.

## The create_graph Module

The `create_graph` script is used to pre-compute and build the mesh graphs required before training any models. This ensures that the complex spatial structures are generated once and loaded efficiently during training.

Reference: {py:mod}`neural_lam.create_graph`

## Graph Types

### Flat Mesh
- Single level of mesh nodes
- Used by GraphLAM

### Hierarchical Mesh
- Multiple levels of mesh nodes at increasing spatial scales
- Used by HiLAM and HiLAMParallel

```{mermaid}
graph TD
    subgraph grid ["Grid Level"]
        G1["Grid Node"] --- G2["Grid Node"] --- G3["Grid Node"]
    end
    subgraph mesh1 ["Mesh Level 1"]
        M1["Mesh Node"] --- M2["Mesh Node"]
    end
    subgraph mesh2 ["Mesh Level 2"]
        M3["Mesh Node"]
    end
    G1 -.-> M1
    G2 -.-> M1
    G2 -.-> M2
    G3 -.-> M2
    M1 -.-> M3
    M2 -.-> M3
```

## Edge Features

For each edge type in the constructed graph, specific features are computed to assist message passing:
- **Spatial Distance**: The physical distance between connected nodes.
- **Directional Vectors**: Vector representations of the direction between nodes, enabling the GNN to understand flow and gradients (e.g., wind direction).
- **Elevation Differences**: Changes in altitude between nodes, which is critical for orographic effects in weather.

## Usage

```bash
python -m neural_lam.create_graph \
    --config_path <config.yaml> \
    --name <graph_name>
```

- `--config_path`: Path to the YAML configuration file which dictates the dataset properties and graph parameters.
- `--name`: The name assigned to the generated graph, which is used to load it during training.

## GNN Layers

The `InteractionNet` represents the core GNN layer implementation in Neural-LAM. It utilizes PyTorch Geometric's `MessagePassing` interface to aggregate features from neighboring nodes and update node states, incorporating edge features natively to refine the spatial interactions.

Reference: {py:mod}`neural_lam.gnn_layers`
