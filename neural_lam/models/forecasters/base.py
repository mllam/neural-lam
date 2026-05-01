# Standard library
from abc import ABC, abstractmethod
from typing import Optional

# Third-party
import torch
from torch import nn


class Forecaster(nn.Module, ABC):
    """
    Generic forecaster capable of mapping from a set of initial states,
    forcing and forces and previous states into a full forecast of the
    requested length.
    """

    @property
    @abstractmethod
    def predicts_std(self) -> bool:
        """Whether this forecaster outputs a predicted standard deviation."""

    @abstractmethod
    def forward(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        boundary_states: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Produce a forecast of length ``pred_steps`` from two initial states,
        the per-step forcing features, and the per-step true boundary states.

        Parameters
        ----------
        init_states : torch.Tensor
            Shape ``(B, 2, num_grid_nodes, d_f)``. The two initial states
            ``[X_{t-1}, X_t]`` used to seed the forecast. Dims: ``B`` is
            batch size, ``2`` is the time index (``[X_{t-1}, X_t]``),
            ``num_grid_nodes`` is the number of spatial nodes, and ``d_f``
            is the state feature dimension.
        forcing_features : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, d_static_f)``. External
            forcings provided at each predicted step. Dims: ``B`` is batch
            size, ``pred_steps`` is the autoregressive rollout length,
            ``num_grid_nodes`` is the number of spatial nodes, and
            ``d_static_f`` is the forcing feature dimension (already
            concatenated past/current/future windows).
        boundary_states : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, d_f)``. True state
            values used ONLY to overwrite boundary nodes at each AR step
            — interior predictions must not depend on ``boundary_states``
            in any other way. This is a temporary mechanism that mirrors
            the pre-refactor ARModel behavior; it will be replaced by a
            dedicated boundary-forcing input in #138 (training on interior
            + boundary datastore), at which point this parameter will be
            removed.

        Returns
        -------
        prediction : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, d_f)``. Forecast of
            state at each predicted step.
        pred_std : torch.Tensor or None
            Shape ``(B, pred_steps, num_grid_nodes, d_f)`` when
            ``predicts_std`` is True, otherwise ``None``. Per-feature
            predicted standard deviation; when ``None``, the constant
            per-variable std is substituted upstream by
            ``ForecasterModule``.
        """
        pass
