"""Latent encoder for flat (non-hierarchical) graphs."""

# Standard library
from typing import Any

# Third-party
import torch

# First-party
from neural_lam import utils
from neural_lam.gnn_layers import get_gnn_class

# Local
from .base_encoder import BaseLatentEncoder


class GraphLatentEncoder(BaseLatentEncoder):
    """
    Latent encoder that maps grid features to mesh and outputs a Gaussian
    distribution over a latent variable on mesh nodes. Uses a flat
    (non-hierarchical) graph: one g2m GNN (type set by ``g2m_gnn_type``)
    followed by a stack of on-mesh (m2m) InteractionNet layers.
    """

    def __init__(
        self,
        latent_dim: int,
        g2m_edge_index: torch.Tensor,
        m2m_edge_index: torch.Tensor,
        hidden_dim: int,
        m2m_layers: int,
        hidden_layers: int = 1,
        g2m_gnn_type: str = "InteractionNet",
        output_dist: str = "isotropic",
    ) -> None:
        """
        Set up the g2m GNN, on-mesh processing stack and latent param map.

        Parameters
        ----------
        latent_dim : int
            Dimensionality of the latent variable at each mesh node.
        g2m_edge_index : torch.Tensor
            Shape ``(2, num_edges_g2m)``. Edge index of grid-to-mesh edges.
        m2m_edge_index : torch.Tensor
            Shape ``(2, num_edges_m2m)``. Edge index of mesh-to-mesh edges.
        hidden_dim : int
            Dimensionality of internal node and edge representations.
        m2m_layers : int
            Number of on-mesh (m2m) GNN layers; 0 disables on-mesh
            processing.
        hidden_layers : int
            Number of hidden layers in internal MLPs.
        g2m_gnn_type : str
            GNN type for the grid-to-mesh step (key in
            ``gnn_layers.GNN_TYPES``).
        output_dist : str
            Type of output distribution: ``"isotropic"`` or ``"diagonal"``.
        """
        super().__init__(latent_dim, output_dist)

        self.g2m_gnn = get_gnn_class(g2m_gnn_type)(
            g2m_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )

        # None if m2m_layers == 0, in which case no on-mesh processing is
        # done in compute_dist_params
        self.m2m_gnns = (
            utils.make_gnn_seq(
                m2m_edge_index, m2m_layers, hidden_layers, hidden_dim
            )
            if m2m_layers > 0
            else None
        )

        self.latent_param_map = utils.make_mlp(
            [hidden_dim] * (hidden_layers + 1) + [self.output_dim],
            layer_norm=False,
        )

    # pylint: disable-next=arguments-differ
    def compute_dist_params(  # type: ignore[override]
        self,
        grid_rep: torch.Tensor,
        graph_emb: dict[str, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute distribution parameters on mesh from grid features.

        Parameters
        ----------
        grid_rep : torch.Tensor
            Shape ``(B, num_grid_nodes, hidden_dim)``. Grid input
            representation.
        graph_emb : dict
            Embedded graph node and edge features, with at least entries
            ``mesh``: ``(B, num_mesh_nodes, hidden_dim)``,
            ``g2m``: ``(B, num_edges_g2m, hidden_dim)`` and
            ``m2m``: ``(B, num_edges_m2m, hidden_dim)``.
        **kwargs
            Ignored.

        Returns
        -------
        torch.Tensor
            Shape ``(B, num_mesh_nodes, output_dim)``. Raw parameters of
            the latent distribution.
        """
        mesh_rep = self.g2m_gnn(grid_rep, graph_emb["mesh"], graph_emb["g2m"])
        if self.m2m_gnns is not None:
            mesh_rep, _ = self.m2m_gnns(mesh_rep, graph_emb["m2m"])
        return self.latent_param_map(mesh_rep)
