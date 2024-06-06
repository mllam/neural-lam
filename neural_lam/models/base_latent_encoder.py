# Third-party
import torch
from torch import distributions as tdists
from torch import nn


class BaseLatentEncoder(nn.Module):
    """
    Abstract class for encoder that maps input to distribution
    over latent variable
    """

    def __init__(
        self,
        latent_dim,
        output_dist="isotropic",
    ):
        super().__init__()

        # Mapping to parameters of latent distribution
        self.output_dist = output_dist
        if output_dist == "isotropic":
            # Isotopic Gaussian, output only mean (\Sigma = I)
            self.output_dim = latent_dim
        elif output_dist == "diagonal":
            # Isotopic Gaussian, output mean and std
            self.output_dim = 2 * latent_dim

            # Small epsilon to prevent enccoding to dist. with std.-dev. 0
            self.latent_std_eps = 1e-4
        else:
            assert False, f"Unknown encoder output distribution: {output_dist}"

    def compute_dist_params(self, grid_rep, **kwargs):
        """
        Compute parameters of distribution over latent variable using the
        grid representation

        grid_rep: (B, num_grid_nodes, d_h)

        Returns:
        parameters: (B, num_mesh_nodes, d_output)
        """
        raise NotImplementedError("compute_dist_params not implemented")

    def forward(self, grid_rep, **kwargs):
        """
        Compute distribution over latent variable

        grid_rep: (B, N_grid, d_h)
        mesh_rep: (B, N_mesh, d_h)
        g2m_rep: (B, M_g2m, d_h)

        Returns:
        distribution: latent var. dist. shaped (B, N_mesh, d_latent)
        """
        latent_dist_params = self.compute_dist_params(grid_rep, **kwargs)

        if self.output_dist == "diagonal":
            latent_mean, latent_std_raw = latent_dist_params.chunk(
                2, dim=-1
            )  # (B, N_mesh, d_latent) and (B, N_mesh, d_latent)
            # pylint: disable-next=not-callable
            latent_std = self.latent_std_eps + nn.functional.softplus(
                latent_std_raw
            )  # positive std.
        else:
            # isotropic
            latent_mean = latent_dist_params
            latent_std = torch.ones_like(latent_mean)

        return tdists.Normal(latent_mean, latent_std)
