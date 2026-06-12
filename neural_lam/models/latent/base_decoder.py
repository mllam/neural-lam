"""Abstract base class for graph-based latent decoders."""

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
        """
        Set up the latent embedder, grid-residual MLP and output param map.

        Parameters
        ----------
        hidden_dim : int
            Dimensionality of internal node and edge representations.
        latent_dim : int
            Dimensionality of the latent variable at each mesh node.
        num_state_vars : int
            Number of state variables predicted at each grid node.
        hidden_layers : int
            Number of hidden layers in the internal MLPs.
        output_std : bool
            If True, the decoder outputs both mean and std of the next-state
            distribution; if False, only the mean.
        """
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

        Parameters
        ----------
        original_grid_rep : torch.Tensor
            Shape ``(B, num_grid_nodes, d_h)``. Grid representation.
        latent_rep : torch.Tensor
            Shape ``(B, num_mesh_nodes, d_h)``. Embedded latent sample.
        residual_grid_rep : torch.Tensor
            Shape ``(B, num_grid_nodes, d_h)``. Grid representation to use
            for residual connections.
        graph_emb : dict
            Embedded graph node and edge features.

        Returns
        -------
        torch.Tensor
            Shape ``(B, num_grid_nodes, d_h)``. Combined grid
            representation.
        """
        raise NotImplementedError("combine_with_latent not implemented")

    def forward(self, grid_rep, latent_samples, last_state, graph_emb):
        """
        Predict mean (and optionally std) of the next weather state.

        Parameters
        ----------
        grid_rep : torch.Tensor
            Shape ``(B, num_grid_nodes, d_h)``. Grid input representation.
        latent_samples : torch.Tensor
            Shape ``(B, num_mesh_nodes, latent_dim)``. Sample of the
            latent variable.
        last_state : torch.Tensor
            Shape ``(B, num_grid_nodes, num_state_vars)``. State at the
            current time step, used as the base of the residual
            prediction.
        graph_emb : dict
            Embedded graph node and edge features, with at least ``g2m``,
            ``m2m`` and ``m2g`` entries.

        Returns
        -------
        pred_mean : torch.Tensor
            Shape ``(B, num_grid_nodes, num_state_vars)``. Predicted mean
            of the next state.
        pred_std : torch.Tensor or None
            Shape ``(B, num_grid_nodes, num_state_vars)`` when
            ``output_std`` is True, otherwise None. Predicted std of the
            next state.
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
