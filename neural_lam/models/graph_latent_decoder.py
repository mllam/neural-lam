# Third-party
import torch_geometric as pyg

# First-party
from neural_lam.interaction_net import InteractionNet, PropagationNet
from neural_lam.models.base_graph_latent_decoder import BaseGraphLatentDecoder


class GraphLatentDecoder(BaseGraphLatentDecoder):
    """
    Decoder that maps grid input + latent variable on mesh to prediction on grid
    Uses non-hierarchical graph
    """

    def __init__(
        self,
        g2m_edge_index,
        m2m_edge_index,
        m2g_edge_index,
        hidden_dim,
        latent_dim,
        processor_layers,
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

        # Processor layers on mesh
        self.processor = pyg.nn.Sequential(
            "mesh_rep, edge_rep",
            [
                (
                    InteractionNet(
                        m2m_edge_index, hidden_dim, hidden_layers=hidden_layers
                    ),
                    "mesh_rep, mesh_rep, edge_rep -> mesh_rep, edge_rep",
                )
                for _ in range(processor_layers)
            ],
        )

        # GNN from mesh to grid
        self.m2g_gnn = PropagationNet(
            m2g_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
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
        mesh_rep = self.g2m_gnn(
            original_grid_rep, latent_rep, graph_emb["g2m"]
        )  # (B, N_mesh, d_h)

        # Process on mesh
        mesh_rep, _ = self.processor(
            mesh_rep, graph_emb["m2m"]
        )  # (B, N_mesh, d_h)

        # Back to grid
        grid_rep = self.m2g_gnn(
            mesh_rep, residual_grid_rep, graph_emb["m2g"]
        )  # (B, N_mesh, d_h)

        return grid_rep
