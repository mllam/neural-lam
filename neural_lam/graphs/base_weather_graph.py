# Standard library
import os

# Third-party
import torch
import torch.nn as nn


class BaseWeatherGraph(nn.Module):
    """
    Graph object representing weather graph consisting of grid and mesh nodes
    """

    def __init__(
        self,
        g2m_edge_index,
        g2m_edge_features,
        m2g_edge_index,
        m2g_edge_features,
    ):
        """
        Create a new graph from tensors
        """
        super().__init__()

        # Store edge indices
        self.g2m_edge_index = g2m_edge_index
        self.m2g_edge_index = m2g_edge_index

        # Store edge features
        self.g2m_edge_features = g2m_edge_features
        self.m2g_edge_features = m2g_edge_features

        # TODO Checks that node indices align
        # TODO Make all node indices start at 0

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

        # Check edge_index
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

        edge_features = BaseWeatherGraph._load_feature_tensor(
            graph_dir_path, f"{subgraph_name}_features.pt"
        )

        # Check compatibility
        assert edge_features.shape[0] == edge_index.shape[1], (
            f"Mismatch in shape of {subgraph_name} edge_index "
            f"(edge_index.shape) and features {edge_features.shape}"
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

        # Check features
        assert features.dtype == torch.float32, (
            f"Wrong data type for {file_name} graph feature tensor: "
            f"{features.dtype}"
        )
        assert len(features.shape) == 2, (
            f"Wrong shape of {file_name} graph feature tensor: "
            f"{features.shape}"
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
