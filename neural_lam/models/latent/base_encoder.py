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

        grid_rep: (B, num_grid_nodes, d_h)

        Returns:
        parameters: (B, num_mesh_nodes, output_dim)
        """
        raise NotImplementedError("compute_dist_params not implemented")

    def forward(self, grid_rep, **kwargs):
        """
        Compute the Gaussian distribution over the latent variable.

        grid_rep: (B, num_grid_nodes, d_h)

        Returns:
        distribution: ``torch.distributions.Normal`` of shape
            (B, num_mesh_nodes, latent_dim)
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
