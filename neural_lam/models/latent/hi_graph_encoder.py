# Third-party
from torch import nn

# First-party
from neural_lam import utils
from neural_lam.gnn_layers import get_gnn_class

# Local
from .base_encoder import BaseLatentEncoder


class HiGraphLatentEncoder(BaseLatentEncoder):
    """
    Latent encoder for a hierarchical mesh: grid -> bottom mesh level via a
    g2m GNN (type set by ``g2m_gnn_type``), then propagates upward through
    mesh levels using mesh-up GNNs (type set by ``mesh_up_gnn_type``), with
    optional intra-level processing at each level. The latent distribution
    is read out from the top mesh level.
    """

    def __init__(
        self,
        latent_dim,
        g2m_edge_index,
        m2m_edge_index,
        mesh_up_edge_index,
        hidden_dim,
        intra_level_layers,
        hidden_layers=1,
        g2m_gnn_type="PropagationNet",
        mesh_up_gnn_type="PropagationNet",
        output_dist="isotropic",
    ):
        super().__init__(latent_dim, output_dist)

        # Hierarchical encoder needs at least 2 mesh levels; with a single
        # level there is no upward propagation and the latent readout would
        # collapse to a flat encoder. Use GraphLatentEncoder instead.
        if len(m2m_edge_index) < 2:
            raise ValueError(
                "HiGraphLatentEncoder requires at least 2 mesh levels "
                f"(got {len(m2m_edge_index)}). Use GraphLatentEncoder for "
                "flat graphs."
            )

        self.g2m_gnn = get_gnn_class(g2m_gnn_type)(
            g2m_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )

        mesh_up_class = get_gnn_class(mesh_up_gnn_type)
        self.mesh_up_gnns = nn.ModuleList(
            [
                mesh_up_class(
                    edge_index,
                    hidden_dim,
                    hidden_layers=hidden_layers,
                    update_edges=False,
                )
                for edge_index in mesh_up_edge_index
            ]
        )

        # Identity mappings if intra_level_layers == 0
        self.intra_level_gnns = nn.ModuleList(
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

        self.latent_param_map = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 1) + [self.output_dim],
            layer_norm=False,
        )

    # pylint: disable-next=arguments-differ
    def compute_dist_params(self, grid_rep, graph_emb, **kwargs):
        """
        Compute distribution parameters on the top mesh level.

        grid_rep: (B, num_grid_nodes, d_h)
        graph_emb: dict with at least
            - ``mesh``: list of (B, num_mesh_nodes[l], d_h)
            - ``g2m``: (B, M_g2m, d_h)
            - ``m2m``: list of (B, M_m2m[l], d_h)
            - ``mesh_up``: list of (B, M_up[l], d_h)

        Returns:
        parameters: (B, num_mesh_nodes[L], output_dim)
        """
        current_mesh_rep = self.g2m_gnn(
            grid_rep, graph_emb["mesh"][0], graph_emb["g2m"]
        )

        # Same-level processing on level 0
        current_mesh_rep, _ = self.intra_level_gnns[0](
            current_mesh_rep, graph_emb["m2m"][0]
        )

        # Walk up levels 1..L
        for (
            up_gnn,
            intra_gnn_seq,
            mesh_up_level_rep,
            m2m_level_rep,
            mesh_level_rep,
        ) in zip(
            self.mesh_up_gnns,
            self.intra_level_gnns[1:],
            graph_emb["mesh_up"],
            graph_emb["m2m"][1:],
            graph_emb["mesh"][1:],
        ):
            new_node_rep = up_gnn(
                current_mesh_rep, mesh_level_rep, mesh_up_level_rep
            )
            current_mesh_rep, _ = intra_gnn_seq(new_node_rep, m2m_level_rep)

        return self.latent_param_map(current_mesh_rep)
