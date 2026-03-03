# Standard library
from abc import ABC, abstractmethod

# Third-party
import torch
from torch import nn


class Forecaster(nn.Module, ABC):
    """
    Generic forecaster capable of mapping from a set of initial states,
    forcing and forces and previous states into a full forecast of the
    requested length.
    """

    @abstractmethod
    def forward(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        boundary_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        init_states: (B, 2, num_grid_nodes, d_f)
        forcing_features: (B, pred_steps, num_grid_nodes, d_static_f)
        boundary_states: (B, pred_steps, num_grid_nodes, d_f)
        Returns:
            prediction: (B, pred_steps, num_grid_nodes, d_f)
            pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)
        """
        pass
