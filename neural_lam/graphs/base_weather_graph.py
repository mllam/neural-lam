# Standard library
import os
from dataclasses import dataclass

# Third-party
import torch
import torch.nn as nn


@dataclass
class BaseWeatherGraph(nn.Module):
    """
    Graph object representing weather graph consisting of grid and mesh nodes
    """

    g2m_edge_index: torch.Tensor
    g2m_edge_features: torch.Tensor
    m2g_edge_index: torch.Tensor
    m2g_edge_features: torch.Tensor

    def __post_init__(self):
        BaseWeatherGraph.check_subgraph(
            self.g2m_edge_features, self.g2m_edge_index, "g2m"
        )
        BaseWeatherGraph.check_subgraph(
            self.m2g_edge_features, self.m2g_edge_index, "m2g"
        )

        # TODO Checks that node indices align
        # TODO Make all node indices start at 0

    @staticmethod
    def check_features(features, subgraph_name):
        """
        Check that feature tensor has the correct format

        features: (2, num_features) tensor of features
        subgraph_name: name of associated subgraph, used in error messages
        """
        assert isinstance(
            features, torch.Tensor
        ), f"{subgraph_name} features is not a tensor"
        assert features.dtype == torch.float32, (
            f"Wrong data type for {subgraph_name} feature tensor: "
            f"{features.dtype}"
        )
        assert len(features.shape) == 2, (
            f"Wrong shape of {subgraph_name} feature tensor: "
            f"{features.shape}"
        )

    @staticmethod
    def check_edge_index(edge_index, subgraph_name):
        """
        Check that edge index tensor has the correct format

        edge_index: (2, num_edges) tensor with edge index
        subgraph_name: name of associated subgraph, used in error messages
        """
        assert isinstance(
            edge_index, torch.Tensor
        ), f"{subgraph_name} edge_index is not a tensor"
        assert edge_index.dtype == torch.int64, (
            f"Wrong data type for {subgraph_name} edge_index "
            f"tensor: {edge_index.dtype}"
        )
        assert len(edge_index.shape) == 2, (
            f"Wrong shape of {subgraph_name} edge_index tensor: "
            f"{edge_index.shape}"
        )
        assert edge_index.shape[0] == 2, (
            "Wrong shape of {subgraph_name} edge_index tensor: "
            f"{edge_index.shape}"
        )

    @staticmethod
    def check_subgraph(edge_features, edge_index, subgraph_name):
        """
        Check that tensors associated with subgraph (edge index and features)
        has the correct format

        edge_features: (2, num_features) tensor of edge features
        edge_index: (2, num_edges) tensor with edge index
        subgraph_name: name of associated subgraph, used in error messages
        """
        # Check individual tensors
        BaseWeatherGraph.check_features(edge_features, subgraph_name)
        BaseWeatherGraph.check_edge_index(edge_index, subgraph_name)

        # Check compatibility
        assert edge_features.shape[0] == edge_index.shape[1], (
            f"Mismatch in shape of {subgraph_name} edge_index "
            f"(edge_index.shape) and features {edge_features.shape}"
        )

    def num_mesh_nodes(self):
        # TODO use g2m
        pass

    @staticmethod
    def from_graph_dir(path):
        """
        Create WeatherGraph from tensors stored in a graph directory

        path: str, path to directory where graph data is stored
        """
        (
            g2m_edge_index,
            g2m_edge_features,
        ) = BaseWeatherGraph._load_subgraph_from_dir(
            path, "g2m"
        )  # (2, M_g2m), (M_g2m, d_edge_features)
        (
            m2g_edge_index,
            m2g_edge_features,
        ) = BaseWeatherGraph._load_subgraph_from_dir(
            path, "m2g"
        )  # (2, M_m2g), (M_m2g, d_edge_features)

        return BaseWeatherGraph(
            g2m_edge_index,
            g2m_edge_features,
            m2g_edge_index,
            m2g_edge_features,
        )

    @staticmethod
    def _load_subgraph_from_dir(graph_dir_path, subgraph_name):
        """
        Load edge_index + feature tensor from a graph directory,
        for a specific subgraph
        """
        edge_index = BaseWeatherGraph._load_graph_tensor(
            graph_dir_path, f"{subgraph_name}_edge_index.pt"
        )

        edge_features = BaseWeatherGraph._load_feature_tensor(
            graph_dir_path, f"{subgraph_name}_features.pt"
        )

        return edge_index, edge_features

    @staticmethod
    def _load_feature_tensor(graph_dir_path, file_name):
        """
        Load feature tensor with from a graph directory
        """
        features = BaseWeatherGraph._load_graph_tensor(
            graph_dir_path, file_name
        )

        return features

    @staticmethod
    def _load_graph_tensor(graph_dir_path, file_name):
        """
        Load graph tensor with edge_index or features from a graph directory
        """
        return torch.load(os.path.join(graph_dir_path, file_name))

    def __str__():
        # TODO Get from graph model init functions
        pass
