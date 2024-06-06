# Third-party
from torch import nn

# First-party
from neural_lam import utils
from neural_lam.interaction_net import InteractionNet, PropagationNet
from neural_lam.models.base_graph_latent_decoder import BaseGraphLatentDecoder


class HiGraphLatentDecoder(BaseGraphLatentDecoder):
    """
    Decoder that maps grid input + latent variable on mesh to prediction on grid
    Uses hierarchical graph
    """

    def __init__(
        self,
        g2m_edge_index,
        m2m_edge_index,
        m2g_edge_index,
        mesh_up_edge_index,
        mesh_down_edge_index,
        hidden_dim,
        latent_dim,
        intra_level_layers,
        hidden_layers=1,
        output_std=True,
    ):
        super().__init__(hidden_dim, latent_dim, hidden_layers, output_std)

        # GNN from grid to mesh
        self.g2m_gnn = InteractionNet(
            g2m_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )
        # GNN from mesh to grid
        self.m2g_gnn = PropagationNet(
            m2g_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )

        # GNNs going up through mesh levels
        self.mesh_up_gnns = nn.ModuleList(
            [
                # Note: We keep these as InteractionNets
                InteractionNet(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                    update_edges=False,
                )
                for edge_index in mesh_up_edge_index
            ]
        )
        # GNNs going down through mesh levels
        self.mesh_down_gnns = nn.ModuleList(
            [
                PropagationNet(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                    update_edges=False,
                )
                for edge_index in mesh_down_edge_index
            ]
        )
        # GNNs applied on intra-level in-between up and down propagation
        # Identity mappings if intra_level_layers = 0
        self.intra_up_gnns = nn.ModuleList(
            [
                utils.make_gnn_seq(
                    edge_index, intra_level_layers, hidden_layers, hidden_dim
                )
                for edge_index in m2m_edge_index
            ]
        )
        self.intra_down_gnns = nn.ModuleList(
            [
                utils.make_gnn_seq(
                    edge_index, intra_level_layers, hidden_layers, hidden_dim
                )
                for edge_index in list(m2m_edge_index)[:-1]
                # Not needed for level L
            ]
        )

    def combine_with_latent(
        self, original_grid_rep, latent_rep, residual_grid_rep, graph_emb
    ):
        """
        Combine the grid representation with representation of latent variable.
        The output should be on the grid again.

        original_grid_rep: (B, num_grid_nodes, d_h)
        latent_rep: (B, num_mesh_nodes, d_h)
        residual_grid_rep: (B, num_grid_nodes, d_h)

        Returns:
        grid_rep: (B, num_grid_nodes, d_h)
        """
        # Map to bottom mesh level
        current_mesh_rep = self.g2m_gnn(
            original_grid_rep, graph_emb["mesh"][0], graph_emb["g2m"]
        )  # (B, num_mesh_nodes[0], d_h)

        # Up hierarchy
        # Run intra-level processing before propagating up
        mesh_level_reps = []
        m2m_level_reps = []
        for (
            up_gnn,
            intra_gnn_seq,
            mesh_up_level_rep,
            m2m_level_rep,
            mesh_level_rep,
        ) in zip(
            self.mesh_up_gnns,
            self.intra_up_gnns[:-1],
            graph_emb["mesh_up"],
            graph_emb["m2m"][:-1],
            # Last propagation up combines with latent representation
            graph_emb["mesh"][1:-1] + [latent_rep],
        ):  # Loop goes L-1 times, from intra-level processing at l=1 to l=L-1
            # Run intra-level processing on level l
            new_mesh_rep, new_m2m_rep = intra_gnn_seq(
                current_mesh_rep, m2m_level_rep
            )  # (B, num_mesh_nodes[l], d_h)

            # Store representation for this level for downward pass
            mesh_level_reps.append(new_mesh_rep)  # Will append L-1 times
            m2m_level_reps.append(new_m2m_rep)

            # Apply up GNN, don't need to store these reps.
            current_mesh_rep = up_gnn(
                new_mesh_rep, mesh_level_rep, mesh_up_level_rep
            )  # (B, num_mesh_nodes[l], d_h)

        # Run intra-level processing for highest mesh level
        current_mesh_rep, _ = self.intra_up_gnns[-1](
            current_mesh_rep, graph_emb["m2m"][-1]
        )  # (B, num_mesh_nodes[L], d_h)

        # Down hierarchy
        # Propagate down before running intra-level processing
        for (
            down_gnn,
            intra_gnn_seq,
            mesh_down_level_rep,
            m2m_level_rep,
            mesh_level_rep,
        ) in zip(
            reversed(self.mesh_down_gnns),
            reversed(self.intra_down_gnns),
            reversed(graph_emb["mesh_down"]),
            reversed(m2m_level_reps),  # Residual connections to up pass
            reversed(mesh_level_reps),  # ^
        ):  # Loop goes L-1 times, from intra level processing at l=L-1 to l=1
            # Apply down GNN, don't need to store these reps.
            new_mesh_rep = down_gnn(
                current_mesh_rep, mesh_level_rep, mesh_down_level_rep
            )  # (B, num_mesh_nodes[l], d_h)

            # Run same level processing on level l
            current_mesh_rep, _ = intra_gnn_seq(
                new_mesh_rep, m2m_level_rep
            )  # (B, num_mesh_nodes[l], d_h)

        # Map back to grid
        grid_rep = self.m2g_gnn(
            current_mesh_rep, residual_grid_rep, graph_emb["m2g"]
        )  # (B, num_mesh_nodes[0], d_h)

        return grid_rep
