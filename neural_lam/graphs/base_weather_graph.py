# Standard library
import functools
import os
from dataclasses import dataclass

# Third-party
import torch
from torch import nn


@dataclass(frozen=True)
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

        # Make all node indices start at 0, if not
        # Use of setattr hereduring initialization, as dataclass is frozen.
        # This matches dataclass behavior used in generated __init__
        # https://docs.python.org/3/library/dataclasses.html#frozen-instances
        # TODO Remove reindexing from from Inets
        object.__setattr__(
            self,
            "g2m_edge_index",
            self._reindex_edge_index(self.g2m_edge_index),
        )
        object.__setattr__(
            self,
            "m2g_edge_index",
            self._reindex_edge_index(self.m2g_edge_index),
        )

    @staticmethod
    def _reindex_edge_index(edge_index):
        """
        Create a version of edge_index with both sender and receiver indices
        starting at 0.

        edge_index: (2, num_edges) tensor with edge index
        """
        return edge_index - edge_index.min(dim=1, keepdim=True)[0]

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

        # TODO Checks that node indices align between edge_index and features

    @functools.cached_property
    def num_grid_nodes(self):
        """
        Get the number of grid nodes (grid cells) that the graph
        is constructed for.
        """
        # Assumes all grid nodes connected to grid
        return self.g2m_edge_index[0].max().item() + 1

    @functools.cached_property
    def num_mesh_nodes(self):
        """
        Get the number of nodes in the mesh graph
        """
        # Assumes all mesh nodes connected to grid
        return self.g2m_edge_index[1].max().item() + 1

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
