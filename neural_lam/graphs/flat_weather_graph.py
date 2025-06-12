# Standard library
import functools
from dataclasses import dataclass

# Third-party
import torch
import torch_geometric as pyg

# First-party
from neural_lam.graphs.base_weather_graph import BaseWeatherGraph


@dataclass(frozen=True)
class FlatWeatherGraph(BaseWeatherGraph):
    """
    Graph object representing weather graph consisting of grid and mesh nodes
    """

    m2m_edge_index: torch.Tensor
    m2m_edge_features: torch.Tensor
    mesh_node_features: torch.Tensor

    def __post_init__(self):
        super().__post_init__()
        BaseWeatherGraph.check_subgraph(
            self.m2m_edge_features, self.m2m_edge_index, "m2m"
        )
        BaseWeatherGraph.check_features(self.mesh_node_features, "mesh nodes")

        # Check that m2m has correct properties
        assert not pyg.utils.contains_isolated_nodes(
            self.m2m_edge_index
        ), "m2m_edge_index does not specify a connected graph"

        # Check that number of mesh nodes is consistent in node and edge sets
        g2m_num = self.g2m_edge_index[1].max().item() + 1
        m2g_num = self.m2g_edge_index[0].max().item() + 1
        m2m_num_from = self.m2m_edge_index[0].max().item() + 1
        m2m_num_to = self.m2m_edge_index[1].max().item() + 1
        for edge_num_mesh, edge_description in (
            (g2m_num, "g2m edges"),
            (m2g_num, "m2g edges"),
            (m2m_num_from, "m2m edges (senders)"),
            (m2m_num_to, "m2m edges (receivers)"),
        ):
            assert edge_num_mesh == self.num_mesh_nodes, (
                "Different number of mesh nodes represented by: "
                f"{edge_description} and mesh node features"
            )

    @functools.cached_property
    def num_mesh_nodes(self):
        """
        Get the number of nodes in the mesh graph
        """
        # Override to determine in more robust way
        # No longer assumes all mesh nodes connected to grid
        return self.mesh_node_features.shape[0]

    @functools.cached_property
    def num_m2m_edges(self):
        """
        Get the number of edges in the mesh graph
        """
        return self.m2m_edge_index.shape[1]

    @staticmethod
    def from_graph_dir(path):
        """
        Create WeatherGraph from tensors stored in a graph directory

        path: str, path to directory where graph data is stored
        """
        # Load base grpah (g2m and m2g)
        base_graph = BaseWeatherGraph.from_graph_dir(path)

        # Load m2m
        (
            m2m_edge_index,
            m2m_edge_features,
        ) = BaseWeatherGraph._load_subgraph_from_dir(
            path, "m2m"
        )  # (2, M_m2m), (M_m2m, d_edge_features)

        # Load static mesh node features
        mesh_node_features = BaseWeatherGraph._load_feature_tensor(
            path, "mesh_features.pt"
        )  # (N_mesh, d_node_features)

        # Note: We assume that graph features are already normalized
        # when read from disk
        # TODO ^ actually do this in graph creation

        return FlatWeatherGraph(
            base_graph.g2m_edge_index,
            base_graph.g2m_edge_features,
            base_graph.m2g_edge_index,
            base_graph.m2g_edge_features,
            m2m_edge_index,
            m2m_edge_features,
            mesh_node_features,
        )

    def __str__(self):
        """
        Returns a string representation of the graph, including the total
        number of nodes and the breakdown of nodes and edges in subgraphs.
        """
        return super().__str__() + f"\nm2m with {self.num_m2m_edges} edges"
