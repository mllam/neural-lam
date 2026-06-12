"""Abstract base class for latent encoders."""

# Third-party
import torch
from torch import distributions as tdists
from torch import nn


class BaseLatentEncoder(nn.Module):
    """
    Abstract encoder mapping an input grid representation to a Gaussian
    distribution over a latent variable defined on mesh nodes.

    Subclasses implement :meth:`compute_dist_params`, which returns the raw
    parameters used to build the output distribution. With
    ``output_dist="isotropic"`` only the mean is produced (unit variance);
    with ``output_dist="diagonal"`` both mean and a positive std are output.
    """

    def __init__(self, latent_dim, output_dist="isotropic"):
        """
        Set up output dimensionality for the chosen distribution type.

        Parameters
        ----------
        latent_dim : int
            Dimensionality of the latent variable at each mesh node.
        output_dist : str
            Type of output distribution: ``"isotropic"`` (mean only, unit
            variance) or ``"diagonal"`` (mean and per-dimension std).
        """
        super().__init__()

        self.output_dist = output_dist
        if output_dist == "isotropic":
            self.output_dim = latent_dim
        elif output_dist == "diagonal":
            self.output_dim = 2 * latent_dim
            # Small floor to prevent the encoder from collapsing to std 0
            self.latent_std_eps = 1e-4
        else:
            raise ValueError(
                f"Unknown encoder output distribution: {output_dist}"
            )

    def compute_dist_params(self, grid_rep, **kwargs):
        """
        Compute raw distribution parameters from the grid representation.

        Parameters
        ----------
        grid_rep : torch.Tensor
            Shape ``(B, num_grid_nodes, d_h)``. Grid input representation.
        **kwargs
            Additional inputs used by concrete encoders (e.g. graph
            embeddings).

        Returns
        -------
        torch.Tensor
            Shape ``(B, num_mesh_nodes, output_dim)``. Raw parameters of
            the latent distribution.
        """
        raise NotImplementedError("compute_dist_params not implemented")

    def forward(self, grid_rep, **kwargs):
        """
        Compute the Gaussian distribution over the latent variable.

        Parameters
        ----------
        grid_rep : torch.Tensor
            Shape ``(B, num_grid_nodes, d_h)``. Grid input representation.
        **kwargs
            Additional inputs forwarded to :meth:`compute_dist_params`.

        Returns
        -------
        torch.distributions.Normal
            Distribution over the latent variable, with batch shape
            ``(B, num_mesh_nodes, latent_dim)``.
        """
        latent_dist_params = self.compute_dist_params(grid_rep, **kwargs)

        if self.output_dist == "diagonal":
            latent_mean, latent_std_raw = latent_dist_params.chunk(2, dim=-1)
            latent_std = self.latent_std_eps + nn.functional.softplus(
                latent_std_raw
            )
        else:
            latent_mean = latent_dist_params
            latent_std = torch.ones_like(latent_mean)

        return tdists.Normal(latent_mean, latent_std)
