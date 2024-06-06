# Third-party
from torch import nn

# First-party
from neural_lam import constants, utils


class BaseGraphLatentDecoder(nn.Module):
    """
    Decoder that maps grid input + latent variable on mesh to prediction on grid
    """

    def __init__(
        self,
        hidden_dim,
        latent_dim,
        hidden_layers=1,
        output_std=True,
    ):
        super().__init__()

        # MLP for residual mapping of grid rep.
        self.grid_update_mlp = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 2)
        )

        # Embedder for latent variable
        self.latent_embedder = utils.make_mlp(
            [latent_dim] + [hidden_dim] * (hidden_layers + 1)
        )

        # Either output input-dependent per-grid-node std or
        # use common per-variable std
        self.output_std = output_std
        if self.output_std:
            output_dim = 2 * constants.GRID_STATE_DIM
        else:
            output_dim = constants.GRID_STATE_DIM

        # Mapping to parameters of state distribution
        self.param_map = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 1) + [output_dim], layer_norm=False
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
        residual_grid_rep: (B, num_grid_nodes, d_h)
        """
        raise NotImplementedError("combine_with_latent not implemented")

    def forward(self, grid_rep, latent_samples, last_state, graph_emb):
        """
        Compute prediction (mean and std.-dev.) of next weather state

        grid_rep: (B, num_grid_nodes, d_h)
        latent_samples: (B, N_mesh, d_latent)
        last_state: (B, num_grid_nodes, d_state)
        graph_emb: dict with graph embedding vectors, entries at least
            g2m: (B, M_g2m, d_h)
            m2m: (B, M_g2m, d_h)
            m2g: (B, M_m2g, d_h)

        Returns:
        mean: (B, N_mesh, d_latent), predicted mean
        std: (B, N_mesh, d_latent), predicted std.-dev.
        """
        # To mesh
        latent_emb = self.latent_embedder(latent_samples)  # (B, N_mesh, d_h)

        # Resiudal MLP for grid representation
        residual_grid_rep = grid_rep + self.grid_update_mlp(
            grid_rep
        )  # (B, num_grid_nodes, d_h)

        combined_grid_rep = self.combine_with_latent(
            grid_rep, latent_emb, residual_grid_rep, graph_emb
        )

        state_params = self.param_map(
            combined_grid_rep
        )  # (B, N_mesh, d_state_params)

        if self.output_std:
            mean_delta, std_raw = state_params.chunk(
                2, dim=-1
            )  # (B, num_grid_nodes, d_state),(B, num_grid_nodes, d_state)
            # pylint: disable-next=not-callable
            pred_std = nn.functional.softplus(std_raw)  # positive std.
        else:
            mean_delta = state_params  # (B, num_grid_nodes, d_state)
            pred_std = None

        pred_mean = last_state + mean_delta

        return pred_mean, pred_std
