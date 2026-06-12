"""Latent decoder for hierarchical graphs."""

# Third-party
from torch import nn

# First-party
from neural_lam import utils
from neural_lam.gnn_layers import (
    InteractionNet,
    PropagationNet,
    get_gnn_class,
)

# Local
from .base_decoder import BaseGraphLatentDecoder


class HiGraphLatentDecoder(BaseGraphLatentDecoder):
    """
    Latent decoder for a hierarchical mesh. The grid representation is
    encoded into the bottom mesh level; the message-passing then propagates
    *up* through the hierarchy (mixing in the latent at the top level), then
    *down* through the hierarchy with residual connections back to the
    intra-level reps from the upward pass, and finally maps back to grid
    via an m2g GNN (type set by ``m2g_gnn_type``). The g2m GNN type is set
    by ``g2m_gnn_type``; mesh-up edges always use InteractionNets and
    mesh-down edges always use PropagationNets.
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
        num_state_vars,
        intra_level_layers,
        hidden_layers=1,
        g2m_gnn_type="InteractionNet",
        m2g_gnn_type="InteractionNet",
        output_std=True,
    ):
        """
        Set up the g2m, m2g, mesh-up/-down and intra-level GNNs.

        Parameters
        ----------
        g2m_edge_index : torch.Tensor
            Shape ``(2, M_g2m)``. Edge index of grid-to-mesh edges.
        m2m_edge_index : BufferList
            Per-level edge indices of intra-level mesh edges, each of shape
            ``(2, M_m2m[l])``.
        m2g_edge_index : torch.Tensor
            Shape ``(2, M_m2g)``. Edge index of mesh-to-grid edges.
        mesh_up_edge_index : BufferList
            Per-level edge indices of upward inter-level mesh edges, each of
            shape ``(2, M_up[l])``.
        mesh_down_edge_index : BufferList
            Per-level edge indices of downward inter-level mesh edges, each
            of shape ``(2, M_down[l])``.
        hidden_dim : int
            Dimensionality of internal node and edge representations.
        latent_dim : int
            Dimensionality of the latent variable at each mesh node.
        num_state_vars : int
            Number of state variables predicted at each grid node.
        intra_level_layers : int
            Number of intra-level GNN layers at each mesh level; 0 disables
            intra-level processing.
        hidden_layers : int
            Number of hidden layers in internal MLPs.
        g2m_gnn_type : str
            GNN type for the grid-to-mesh step (key in
            ``gnn_layers.GNN_TYPES``).
        m2g_gnn_type : str
            GNN type for the mesh-to-grid step (key in
            ``gnn_layers.GNN_TYPES``). Inter-level edges are not
            configurable; mesh-up edges always use ``InteractionNet`` and
            mesh-down edges always use ``PropagationNet``.
        output_std : bool
            If True, the decoder outputs both mean and std of the next-state
            distribution; if False, only the mean.
        """
        super().__init__(
            hidden_dim, latent_dim, num_state_vars, hidden_layers, output_std
        )

        # Hierarchical decoder needs at least 2 mesh levels; with a single
        # level the up/down passes are empty and the latent would be
        # silently ignored. Use GraphLatentDecoder instead.
        if len(m2m_edge_index) < 2:
            raise ValueError(
                "HiGraphLatentDecoder requires at least 2 mesh levels "
                f"(got {len(m2m_edge_index)}). Use GraphLatentDecoder for "
                "flat graphs."
            )

        self.g2m_gnn = get_gnn_class(g2m_gnn_type)(
            g2m_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )
        self.m2g_gnn = get_gnn_class(m2g_gnn_type)(
            m2g_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )

        # Mesh-up edges must use InteractionNet: with a PropagationNet the
        # latent rep at the top level would be overwritten rather than
        # residually updated, leaving Z unused at initialization.
        self.mesh_up_gnns = nn.ModuleList(
            [
                InteractionNet(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                    update_edges=False,
                )
                for edge_index in mesh_up_edge_index
            ]
        )
        # Mesh-down edges must use PropagationNet: each downward step has to
        # push the latent information from the level above into the lower
        # level, so that Z reaches the grid output.
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

        # Identity mappings if intra_level_layers == 0
        self.intra_up_gnns = nn.ModuleList(
            [
                (
                    utils.make_gnn_seq(
                        edge_index,
                        intra_level_layers,
                        hidden_layers,
                        hidden_dim,
                    )
                    if intra_level_layers > 0
                    else utils.IdentityModule()
                )
                for edge_index in m2m_edge_index
            ]
        )
        self.intra_down_gnns = nn.ModuleList(
            [
                (
                    utils.make_gnn_seq(
                        edge_index,
                        intra_level_layers,
                        hidden_layers,
                        hidden_dim,
                    )
                    if intra_level_layers > 0
                    else utils.IdentityModule()
                )
                for edge_index in list(m2m_edge_index)[:-1]
                # Top level (L) does not need a down intra-level GNN
            ]
        )

    def combine_with_latent(
        self, original_grid_rep, latent_rep, residual_grid_rep, graph_emb
    ):
        """
        Hierarchical up-then-down fusion of grid and latent reps.

        original_grid_rep: (B, num_grid_nodes, d_h)
        latent_rep: (B, num_mesh_nodes[L], d_h)
        residual_grid_rep: (B, num_grid_nodes, d_h)
        graph_emb: dict with at least
            - ``mesh``: list of (B, num_mesh_nodes[l], d_h)
            - ``g2m``: (B, M_g2m, d_h)
            - ``m2m``: list of (B, M_m2m[l], d_h)
            - ``mesh_up``: list of (B, M_up[l], d_h)
            - ``mesh_down``: list of (B, M_down[l], d_h)
            - ``m2g``: (B, M_m2g, d_h)

        Returns:
        grid_rep: (B, num_grid_nodes, d_h)
        """
        current_mesh_rep = self.g2m_gnn(
            original_grid_rep, graph_emb["mesh"][0], graph_emb["g2m"]
        )

        # Upward pass: intra-level processing, then up to the next level.
        # On the last upward step, the latent replaces the level-L mesh rep
        # so the latent is fused in at the top of the hierarchy.
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
            graph_emb["mesh"][1:-1] + [latent_rep],
        ):
            new_mesh_rep, new_m2m_rep = intra_gnn_seq(
                current_mesh_rep, m2m_level_rep
            )

            mesh_level_reps.append(new_mesh_rep)
            m2m_level_reps.append(new_m2m_rep)

            current_mesh_rep = up_gnn(
                new_mesh_rep, mesh_level_rep, mesh_up_level_rep
            )

        # Top level processing
        current_mesh_rep, _ = self.intra_up_gnns[-1](
            current_mesh_rep, graph_emb["m2m"][-1]
        )

        # Downward pass: down GNN, then intra-level processing. Residual
        # connections feed back the intra-level reps from the upward pass.
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
            reversed(m2m_level_reps),
            reversed(mesh_level_reps),
        ):
            new_mesh_rep = down_gnn(
                current_mesh_rep, mesh_level_rep, mesh_down_level_rep
            )
            current_mesh_rep, _ = intra_gnn_seq(new_mesh_rep, m2m_level_rep)

        grid_rep = self.m2g_gnn(
            current_mesh_rep, residual_grid_rep, graph_emb["m2g"]
        )

        return grid_rep
