# First-party
from neural_lam import utils
from neural_lam.gnn_layers import get_gnn_class

# Local
from .base_encoder import BaseLatentEncoder


class GraphLatentEncoder(BaseLatentEncoder):
    """
    Latent encoder that maps grid features to mesh and outputs a Gaussian
    distribution over a latent variable on mesh nodes. Uses a flat
    (non-hierarchical) graph: one g2m GNN (type set by ``g2m_gnn_type``)
    followed by a stack of on-mesh (m2m) InteractionNet layers.
    """

    def __init__(
        self,
        latent_dim,
        g2m_edge_index,
        m2m_edge_index,
        hidden_dim,
        m2m_layers,
        hidden_layers=1,
        g2m_gnn_type="PropagationNet",
        output_dist="isotropic",
    ):
        super().__init__(latent_dim, output_dist)

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

        self.latent_param_map = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 1) + [self.output_dim],
            layer_norm=False,
        )

    # pylint: disable-next=arguments-differ
    def compute_dist_params(self, grid_rep, graph_emb, **kwargs):
        """
        Compute distribution parameters on mesh from grid features.

        grid_rep: (B, num_grid_nodes, d_h)
        graph_emb: dict with at least
            - ``mesh``: (B, num_mesh_nodes, d_h)
            - ``g2m``: (B, M_g2m, d_h)
            - ``m2m``: (B, M_m2m, d_h)

        Returns:
        parameters: (B, num_mesh_nodes, output_dim)
        """
        mesh_rep = self.g2m_gnn(grid_rep, graph_emb["mesh"], graph_emb["g2m"])
        mesh_rep, _ = self.m2m_gnns(mesh_rep, graph_emb["m2m"])
        return self.latent_param_map(mesh_rep)
