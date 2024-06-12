# First-party
from neural_lam.graphs.base_weather_graph import BaseWeatherGraph


class FlatWeatherGraph(BaseWeatherGraph):
    """
    Graph object representing weather graph consisting of grid and mesh nodes
    """

    def __init__(
        self,
        g2m_edge_index,
        g2m_edge_features,
        m2g_edge_index,
        m2g_edge_features,
        m2m_edge_index,
        m2m_edge_features,
        mesh_node_features,
    ):
        """
        Create a new graph from tensors
        """
        super().__init__(
            g2m_edge_index,
            g2m_edge_features,
            m2g_edge_index,
            m2g_edge_features,
        )

        # Store mesh tensors
        self.m2m_edge_index = m2m_edge_index
        self.m2m_edge_features = m2m_edge_features
        self.mesh_node_features = mesh_node_features

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
