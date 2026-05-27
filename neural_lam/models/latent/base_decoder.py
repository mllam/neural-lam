# Third-party
from torch import nn

# First-party
from neural_lam import utils


class BaseGraphLatentDecoder(nn.Module):
    """
    Abstract decoder mapping a grid representation plus a latent sample on
    mesh to the parameters of the next-state distribution on the grid.

    Subclasses implement :meth:`combine_with_latent`, which fuses the latent
    representation with the grid representation. The resulting features are
    mapped to either ``num_state_vars`` outputs (mean only) or
    ``2 * num_state_vars`` outputs (mean + softplus std) depending on
    ``output_std``.
    """

    def __init__(
        self,
        hidden_dim,
        latent_dim,
        num_state_vars,
        hidden_layers=1,
        output_std=True,
    ):
        super().__init__()

        self.grid_update_mlp = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 2)
        )

        self.latent_embedder = utils.make_mlp(
            [latent_dim] + [hidden_dim] * (hidden_layers + 1)
        )

        self.output_std = output_std
        if self.output_std:
            output_dim = 2 * num_state_vars
        else:
            output_dim = num_state_vars

        self.param_map = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 1) + [output_dim], layer_norm=False
        )

    def combine_with_latent(
        self, original_grid_rep, latent_rep, residual_grid_rep, graph_emb
    ):
        """
        Fuse grid and latent representations and return a grid-shaped output.

        original_grid_rep: (B, num_grid_nodes, d_h)
        latent_rep: (B, num_mesh_nodes, d_h)
        residual_grid_rep: (B, num_grid_nodes, d_h)
        graph_emb: dict of graph edge / node embeddings

        Returns:
        combined_grid_rep: (B, num_grid_nodes, d_h)
        """
        raise NotImplementedError("combine_with_latent not implemented")

    def forward(self, grid_rep, latent_samples, last_state, graph_emb):
        """
        Predict mean (and optionally std) of the next weather state.

        grid_rep: (B, num_grid_nodes, d_h)
        latent_samples: (B, num_mesh_nodes, latent_dim)
        last_state: (B, num_grid_nodes, num_state_vars)
        graph_emb: dict with at least ``g2m``, ``m2m``, ``m2g`` entries

        Returns:
        pred_mean: (B, num_grid_nodes, num_state_vars)
        pred_std: (B, num_grid_nodes, num_state_vars) or ``None``
        """
        latent_emb = self.latent_embedder(latent_samples)

        residual_grid_rep = grid_rep + self.grid_update_mlp(grid_rep)

        combined_grid_rep = self.combine_with_latent(
            grid_rep, latent_emb, residual_grid_rep, graph_emb
        )

        state_params = self.param_map(combined_grid_rep)

        if self.output_std:
            mean_delta, std_raw = state_params.chunk(2, dim=-1)
            pred_std = nn.functional.softplus(std_raw)
        else:
            mean_delta = state_params
            pred_std = None

        pred_mean = last_state + mean_delta

        return pred_mean, pred_std
