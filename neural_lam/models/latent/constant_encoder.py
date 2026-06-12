"""Constant (input-independent) latent encoder."""

# Third-party
import torch

# Local
from .base_encoder import BaseLatentEncoder


class ConstantLatentEncoder(BaseLatentEncoder):
    """
    Latent encoder that returns a constant (input-independent) distribution.

    ``compute_dist_params`` returns a tensor of zeros, so the resulting
    Normal is ``Normal(mean=0, std=1)`` for ``output_dist="isotropic"`` and
    ``Normal(mean=0, std=softplus(0)+eps)`` for ``output_dist="diagonal"``.
    """

    def __init__(self, latent_dim, num_mesh_nodes, output_dist="isotropic"):
        """
        Store the number of mesh nodes to produce parameters for.

        Parameters
        ----------
        latent_dim : int
            Dimensionality of the latent variable at each mesh node.
        num_mesh_nodes : int
            Number of mesh nodes the latent variable is defined on.
        output_dist : str
            Type of output distribution: ``"isotropic"`` or ``"diagonal"``.
        """
        super().__init__(latent_dim, output_dist)
        self.num_mesh_nodes = num_mesh_nodes

    def compute_dist_params(self, grid_rep, **kwargs):
        """
        Compute raw distribution parameters from the grid representation.

        For this constant encoder the parameters are all zeros, independent
        of the values in ``grid_rep``.

        Parameters
        ----------
        grid_rep : torch.Tensor
            Shape ``(B, num_grid_nodes, d_h)``. Grid input representation,
            used only to determine batch size and device.
        **kwargs
            Ignored; accepted for compatibility with the base class
            interface.

        Returns
        -------
        torch.Tensor
            Shape ``(B, num_mesh_nodes, output_dim)``. Raw parameters of
            the latent distribution, all zeros.
        """
        return torch.zeros(
            grid_rep.shape[0],
            self.num_mesh_nodes,
            self.output_dim,
            device=grid_rep.device,
        )
