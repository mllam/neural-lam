# First-party
from neural_lam import utils
from neural_lam.interaction_net import PropagationNet
from neural_lam.models.base_latent_encoder import BaseLatentEncoder


class GraphLatentEncoder(BaseLatentEncoder):
    """
    Encoder that maps from grid to mesh and defines a latent distribution
    on mesh
    """

    def __init__(
        self,
        latent_dim,
        g2m_edge_index,
        m2m_edge_index,
        hidden_dim,
        processor_layers,
        hidden_layers=1,
        output_dist="isotropic",
    ):
        super().__init__(
            latent_dim,
            output_dist,
        )

        # GNN from grid to mesh
        self.g2m_gnn = PropagationNet(
            g2m_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )

        # Processor layers on mesh
        self.processor = utils.make_gnn_seq(
            m2m_edge_index, processor_layers, hidden_layers, hidden_dim
        )

        self.latent_param_map = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 1) + [self.output_dim],
            layer_norm=False,
        )

    # pylint: disable-next=arguments-differ
    def compute_dist_params(self, grid_rep, graph_emb, **kwargs):
        """
        Compute parameters of distribution over latent variable using the
        grid representation

        grid_rep: (B, N_grid, d_h)
        graph_emb: dict with graph embedding vectors, entries at least
            mesh: (B, N_mesh, d_h)
            g2m: (B, M_g2m, d_h)
            m2m: (B, M_g2m, d_h)

        Returns:
        parameters: (B, num_mesh_nodes, d_output)
        """
        mesh_rep = self.g2m_gnn(
            grid_rep, graph_emb["mesh"], graph_emb["g2m"]
        )  # (B, N_mesh, d_h)
        mesh_rep, _ = self.processor(mesh_rep, graph_emb["m2m"])
        return self.latent_param_map(mesh_rep)  # (B, N_mesh, d_output)
