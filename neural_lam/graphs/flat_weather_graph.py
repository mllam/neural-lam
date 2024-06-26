# Standard library
from dataclasses import dataclass

# Third-party
import torch

# First-party
from neural_lam.graphs.base_weather_graph import BaseWeatherGraph


@dataclass
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
        # TODO Checks that node indices align

    def num_mesh_nodes(self):
        # TODO use mesh_node_features
        pass

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
