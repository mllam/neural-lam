# Third-party
from torch import nn

# Local
from .. import utils
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..graph_data import GraphSizes
from ..interaction_net import InteractionNet
from .base_graph_model import BaseGraphModel


class BaseHiGraphModel(BaseGraphModel):
    """
    Base class for hierarchical graph models.
    """

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        graph_sizes: GraphSizes,
    ):
        """
        Parameters
        ----------
        graph_sizes : GraphSizes
            Graph metadata as described in ``BaseGraphModel`` including
            per-level mesh sizes, feature dimensions, and edge counts for
            same-, up-, and down-level connections (``mesh_level_sizes``,
            ``mesh_feature_dims``, ``m2m_feature_dims``,
            ``mesh_up_edge_counts``, ``mesh_down_edge_counts`` with
            corresponding feature dimensions).
        """
        super().__init__(
            args, config=config, datastore=datastore, graph_sizes=graph_sizes
        )

        m2m_edge_counts = self.graph_sizes.m2m_edge_counts
        mesh_up_edge_counts = self.graph_sizes.mesh_up_edge_counts
        mesh_down_edge_counts = self.graph_sizes.mesh_down_edge_counts

        mesh_feature_dims = self.graph_sizes.mesh_feature_dims
        m2m_feature_dims = self.graph_sizes.m2m_feature_dims
        mesh_up_feature_dims = self.graph_sizes.mesh_up_feature_dims
        mesh_down_feature_dims = self.graph_sizes.mesh_down_feature_dims

        # Print some useful info
        utils.rank_zero_print("Loaded hierarchical graph with structure:")
        for level_index, level_mesh_size in enumerate(
            self.graph_sizes.mesh_level_sizes
        ):
            same_level_edges = m2m_edge_counts[level_index]
            utils.rank_zero_print(
                f"level {level_index} - {level_mesh_size} nodes, "
                f"{same_level_edges} same-level edges"
            )

            if level_index < (self.graph_sizes.num_levels - 1):
                up_edges = mesh_up_edge_counts[level_index]
                down_edges = mesh_down_edge_counts[level_index]
                utils.rank_zero_print(f"  {level_index}<->{level_index + 1}")
                utils.rank_zero_print(
                    f" - {up_edges} up edges, {down_edges} down edges"
                )
        # Embedders
        # Assume all levels have same static feature dimensionality
        mesh_dim = mesh_feature_dims[0]
        mesh_same_dim = m2m_feature_dims[0]
        mesh_up_dim = (
            mesh_up_feature_dims[0] if mesh_up_feature_dims else mesh_dim
        )
        mesh_down_dim = (
            mesh_down_feature_dims[0] if mesh_down_feature_dims else mesh_dim
        )

        # Separate mesh node embedders for each level
        self.mesh_embedders = nn.ModuleList(
            [
                utils.make_mlp([mesh_dim] + self.mlp_blueprint_end)
                for _ in range(self.graph_sizes.num_levels)
            ]
        )
        self.mesh_same_embedders = nn.ModuleList(
            [
                utils.make_mlp([mesh_same_dim] + self.mlp_blueprint_end)
                for _ in range(self.graph_sizes.num_levels)
            ]
        )
        self.mesh_up_embedders = nn.ModuleList(
            [
                utils.make_mlp([mesh_up_dim] + self.mlp_blueprint_end)
                for _ in range(self.graph_sizes.num_levels - 1)
            ]
        )
        self.mesh_down_embedders = nn.ModuleList(
            [
                utils.make_mlp([mesh_down_dim] + self.mlp_blueprint_end)
                for _ in range(self.graph_sizes.num_levels - 1)
            ]
        )

        # Instantiate GNNs
        # Init GNNs
        self.mesh_init_gnns = nn.ModuleList(
            [
                InteractionNet(
                    args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                )
                for _ in mesh_up_feature_dims
            ]
        )

        # Read out GNNs
        self.mesh_read_gnns = nn.ModuleList(
            [
                InteractionNet(
                    args.hidden_dim,
                    hidden_layers=args.hidden_layers,
                    update_edges=False,
                )
                for _ in mesh_down_feature_dims
            ]
        )

    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        num_mesh_nodes = self.graph_sizes.num_mesh_nodes
        num_mesh_nodes_ignore = (
            num_mesh_nodes - self.graph_sizes.mesh_level_sizes[0]
        )
        return num_mesh_nodes, num_mesh_nodes_ignore

    def embedd_mesh_nodes(self):
        """
        Embed static mesh features
        This embeds only bottom level, rest is done at beginning of
        processing step
        Returns tensor of shape (num_mesh_nodes[0], d_h)
        """
        if self.current_graph is None:
            raise RuntimeError(
                "Graph data not set before embedding mesh nodes."
            )
        return self.mesh_embedders[0](
            self.current_graph["mesh_static_features"][0]
        )

    def process_step(self, mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, num_mesh_nodes, d_h)
        Returns mesh_rep: (B, num_mesh_nodes, d_h)
        """
        if self.current_graph is None:
            raise RuntimeError("Graph data not set before processing mesh.")

        batch_size = mesh_rep.shape[0]
        mesh_static = self.current_graph["mesh_static_features"]
        mesh_same_features = self.current_graph["m2m_features"]
        mesh_up_features = self.current_graph["mesh_up_features"]
        mesh_down_features = self.current_graph["mesh_down_features"]
        mesh_up_edges = self.current_graph["mesh_up_edge_index"]
        mesh_down_edges = self.current_graph["mesh_down_edge_index"]

        # EMBED REMAINING MESH NODES (levels >= 1) -
        # Create list of mesh node representations for each level,
        # each of size (B, num_mesh_nodes[l], d_h)
        mesh_rep_levels = [mesh_rep] + [
            self.expand_to_batch(emb(node_static_features), batch_size)
            for emb, node_static_features in zip(
                list(self.mesh_embedders)[1:],
                list(mesh_static)[1:],
            )
        ]

        # - EMBED EDGES -
        # Embed edges, expand with batch dimension
        mesh_same_rep = [
            self.expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(
                self.mesh_same_embedders, mesh_same_features
            )
        ]
        mesh_up_rep = [
            self.expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(self.mesh_up_embedders, mesh_up_features)
        ]
        mesh_down_rep = [
            self.expand_to_batch(emb(edge_feat), batch_size)
            for emb, edge_feat in zip(
                self.mesh_down_embedders, mesh_down_features
            )
        ]

        # - MESH INIT. -
        # Let level_l go from 1 to L
        for level_l, gnn in enumerate(self.mesh_init_gnns, start=1):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l - 1
            ]  # (B, num_mesh_nodes[l-1], d_h)
            rec_node_rep = mesh_rep_levels[
                level_l
            ]  # (B, num_mesh_nodes[l], d_h)
            edge_rep = mesh_up_rep[level_l - 1]

            # Apply GNN
            new_node_rep, new_edge_rep = gnn(
                send_node_rep,
                rec_node_rep,
                edge_rep,
                mesh_up_edges[level_l - 1],
            )

            # Update node and edge vectors in lists
            mesh_rep_levels[
                level_l
            ] = new_node_rep  # (B, num_mesh_nodes[l], d_h)
            mesh_up_rep[level_l - 1] = new_edge_rep  # (B, M_up[l-1], d_h)

        # - PROCESSOR -
        mesh_rep_levels, _, _, mesh_down_rep = self.hi_processor_step(
            mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
        )

        # - MESH READ OUT. -
        # Let level_l go from L-1 to 0
        for level_l, gnn in zip(
            range(self.graph_sizes.num_levels - 2, -1, -1),
            reversed(self.mesh_read_gnns),
        ):
            # Extract representations
            send_node_rep = mesh_rep_levels[
                level_l + 1
            ]  # (B, num_mesh_nodes[l+1], d_h)
            rec_node_rep = mesh_rep_levels[
                level_l
            ]  # (B, num_mesh_nodes[l], d_h)
            edge_rep = mesh_down_rep[level_l]

            # Apply GNN
            new_node_rep = gnn(
                send_node_rep,
                rec_node_rep,
                edge_rep,
                mesh_down_edges[level_l],
            )

            # Update node and edge vectors in lists
            mesh_rep_levels[
                level_l
            ] = new_node_rep  # (B, num_mesh_nodes[l], d_h)

        # Return only bottom level representation
        return mesh_rep_levels[0]  # (B, num_mesh_nodes[0], d_h)

    def hi_processor_step(
        self, mesh_rep_levels, mesh_same_rep, mesh_up_rep, mesh_down_rep
    ):
        """
        Internal processor step of hierarchical graph models.
        Between mesh init and read out.

        Each input is list with representations, each with shape

        mesh_rep_levels: (B, num_mesh_nodes[l], d_h)
        mesh_same_rep: (B, M_same[l], d_h)
        mesh_up_rep: (B, M_up[l -> l+1], d_h)
        mesh_down_rep: (B, M_down[l <- l+1], d_h)

        Returns same lists
        """
        raise NotImplementedError("hi_process_step not implemented")
