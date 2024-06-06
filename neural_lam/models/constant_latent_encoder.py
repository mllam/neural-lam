# Third-party
import torch

# First-party
from neural_lam.models.base_latent_encoder import BaseLatentEncoder


class ConstantLatentEncoder(BaseLatentEncoder):
    """
    Latent encoder parametrizing constant distribution
    """

    def __init__(
        self,
        latent_dim,
        num_mesh_nodes,
        output_dist="isotropic",
    ):
        super().__init__(
            latent_dim,
            output_dist,
        )

        self.num_mesh_nodes = num_mesh_nodes

    def compute_dist_params(self, grid_rep, **kwargs):
        """
        Compute parameters of distribution over latent variable using the
        grid representation

        grid_rep: (B, num_grid_nodes, d_h)

        Returns:
        parameters: (B, num_mesh_nodes, d_output)
        """
        return torch.ones(
            grid_rep.shape[0],
            self.num_mesh_nodes,
            self.output_dim,
            device=grid_rep.device,
        )  # (B, num_mesh_nodes, d_output)
