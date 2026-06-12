"""Latent decoder for flat (non-hierarchical) graphs."""

# First-party
from neural_lam import utils
from neural_lam.gnn_layers import get_gnn_class

# Local
from .base_decoder import BaseGraphLatentDecoder


class GraphLatentDecoder(BaseGraphLatentDecoder):
    """
    Latent decoder for a flat (non-hierarchical) graph. Encodes grid into
    mesh with a g2m GNN (type set by ``g2m_gnn_type``), processes on mesh,
    and reads back out to grid with an m2g GNN (type set by ``m2g_gnn_type``).
    The grid representation also goes through a residual MLP that is added
    back to the mesh-to-grid output.
    """

    def __init__(
        self,
        g2m_edge_index,
        m2m_edge_index,
        m2g_edge_index,
        hidden_dim,
        latent_dim,
        num_state_vars,
        m2m_layers,
        hidden_layers=1,
        g2m_gnn_type="InteractionNet",
        m2g_gnn_type="InteractionNet",
        output_std=True,
    ):
        """
        Set up the g2m, on-mesh and m2g GNNs.

        Parameters
        ----------
        g2m_edge_index : torch.Tensor
            Shape ``(2, M_g2m)``. Edge index of grid-to-mesh edges.
        m2m_edge_index : torch.Tensor
            Shape ``(2, M_m2m)``. Edge index of mesh-to-mesh edges.
        m2g_edge_index : torch.Tensor
            Shape ``(2, M_m2g)``. Edge index of mesh-to-grid edges.
        hidden_dim : int
            Dimensionality of internal node and edge representations.
        latent_dim : int
            Dimensionality of the latent variable at each mesh node.
        num_state_vars : int
            Number of state variables predicted at each grid node.
        m2m_layers : int
            Number of on-mesh (m2m) GNN layers; 0 disables on-mesh
            processing.
        hidden_layers : int
            Number of hidden layers in internal MLPs.
        g2m_gnn_type : str
            GNN type for the grid-to-mesh step (key in
            ``gnn_layers.GNN_TYPES``).
        m2g_gnn_type : str
            GNN type for the mesh-to-grid step (key in
            ``gnn_layers.GNN_TYPES``).
        output_std : bool
            If True, the decoder outputs both mean and std of the next-state
            distribution; if False, only the mean.
        """
        super().__init__(
            hidden_dim, latent_dim, num_state_vars, hidden_layers, output_std
        )

        self.g2m_gnn = get_gnn_class(g2m_gnn_type)(
            g2m_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )

        self.m2m_gnns = (
            utils.make_gnn_seq(
                m2m_edge_index, m2m_layers, hidden_layers, hidden_dim
            )
            if m2m_layers > 0
            else utils.IdentityModule()
        )

        self.m2g_gnn = get_gnn_class(m2g_gnn_type)(
            m2g_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )

    def combine_with_latent(
        self, original_grid_rep, latent_rep, residual_grid_rep, graph_emb
    ):
        """
        Fuse grid and latent reps via g2m -> m2m -> m2g.

        original_grid_rep: (B, num_grid_nodes, d_h)
        latent_rep: (B, num_mesh_nodes, d_h)
        residual_grid_rep: (B, num_grid_nodes, d_h)

        Returns:
        grid_rep: (B, num_grid_nodes, d_h)
        """
        mesh_rep = self.g2m_gnn(original_grid_rep, latent_rep, graph_emb["g2m"])

        mesh_rep, _ = self.m2m_gnns(mesh_rep, graph_emb["m2m"])

        grid_rep = self.m2g_gnn(mesh_rep, residual_grid_rep, graph_emb["m2g"])

        return grid_rep
