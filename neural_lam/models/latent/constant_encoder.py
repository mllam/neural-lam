# Third-party
import torch

# Local
from .base_encoder import BaseLatentEncoder


class ConstantLatentEncoder(BaseLatentEncoder):
    """
    Latent encoder that returns a constant (input-independent) distribution.

    Used as a non-learned prior in ``GraphEFM`` when ``learn_prior`` is
    disabled. ``compute_dist_params`` returns a tensor of ones, so the
    resulting Normal is ``Normal(mean=1, std=1)`` for ``output_dist=
    "isotropic"`` and ``Normal(mean=1, std=softplus(1)+eps)`` for
    ``output_dist="diagonal"``. (Note: the ``train_model.py`` CLI help on
    ``prob_model_lam`` describes this prior as "mean 0"; the code itself
    has always produced mean 1. Preserved as-is during the port for
    behavioral parity — open question for upstream.)
    """

    def __init__(self, latent_dim, num_mesh_nodes, output_dist="isotropic"):
        super().__init__(latent_dim, output_dist)
        self.num_mesh_nodes = num_mesh_nodes

    def compute_dist_params(self, grid_rep, **kwargs):
        """
        Return constant parameters of shape (B, num_mesh_nodes, output_dim).
        """
        return torch.ones(
            grid_rep.shape[0],
            self.num_mesh_nodes,
            self.output_dim,
            device=grid_rep.device,
        )
