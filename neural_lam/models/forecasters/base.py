"""Base class for forecasters."""

# Standard library
from abc import ABC, abstractmethod
from typing import Callable

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
        """
        Whether this forecaster outputs a predicted standard deviation.

        Returns
        -------
        bool
            ``True`` if the forecaster predicts standard deviation,
            ``False`` otherwise.
        """

    @abstractmethod
    def forward(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        boundary_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Produce a forecast of length ``pred_steps`` from two initial states,
        the per-step forcing features, and the per-step true boundary states.

        Parameters
        ----------
        init_states : torch.Tensor
            Shape ``(B, 2, num_grid_nodes, num_state_vars)``. The two initial
            states ``[X_{t-1}, X_t]`` used to start the forecast from. Dims:
            ``B`` is batch size, ``2`` is the time index (``[X_{t-1}, X_t]``),
            ``num_grid_nodes`` is the number of spatial nodes, and
            ``num_state_vars`` is the state feature dimension.
        forcing_features : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_forcing_vars)``.
            External forcings provided at each predicted step. Dims: ``B``
            is batch size, ``pred_steps`` is the autoregressive rollout
            length, ``num_grid_nodes`` is the number of spatial nodes, and
            ``num_forcing_vars`` is the forcing feature dimension (already
            concatenated past/current/future windows).
        boundary_states : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``. True
            state values used ONLY to overwrite boundary nodes at each AR
            step; interior predictions must not depend on ``boundary_states``
            in any other way. Dims: ``B`` is batch size, ``pred_steps`` is
            the rollout length, ``num_grid_nodes`` is the number of spatial
            nodes, and ``num_state_vars`` is the state feature dimension.
            This is a temporary mechanism that mirrors the pre-refactor
            ARModel behavior; it will be replaced by a dedicated
            boundary-forcing input in #138 (training on interior + boundary
            datastore), at which point this parameter will be removed.

        Returns
        -------
        prediction : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``.
            Forecast of state at each predicted step. Dims: same as
            ``boundary_states``.
        pred_std : torch.Tensor or None
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)`` when
            ``predicts_std`` is True, otherwise ``None``. Per-feature
            predicted standard deviation; when ``None``, the constant
            per-variable std is substituted upstream by
            ``ForecasterModule``. Dims: same as ``prediction``.
        """

    @abstractmethod
    def compute_training_loss(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        target_states: torch.Tensor,
        score_fn: Callable[..., torch.Tensor],
        interior_mask_bool: torch.Tensor,
        per_var_std: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute the training objective for one batch.

        The forecaster owns its complete training objective: which forecasts
        to produce from the batch, which loss terms to compute from them and
        how to combine those terms into a single scalar. The wrapping
        ``ForecasterModule`` only injects the configured scoring rule and
        mask, logs the returned components and optimizes the returned loss.

        Parameters
        ----------
        init_states : torch.Tensor
            Shape ``(B, 2, num_grid_nodes, num_state_vars)``. The two initial
            states ``[X_{t-1}, X_t]`` used to start the forecast from. Dims:
            ``B`` is batch size, ``2`` is the time index (``[X_{t-1}, X_t]``),
            ``num_grid_nodes`` is the number of spatial nodes, and
            ``num_state_vars`` is the state feature dimension.
        forcing_features : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_forcing_vars)``.
            External forcings provided at each predicted step. Dims: ``B``
            is batch size, ``pred_steps`` is the rollout length,
            ``num_grid_nodes`` is the number of spatial nodes, and
            ``num_forcing_vars`` is the forcing feature dimension (already
            concatenated past/current/future windows).
        target_states : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``. True
            states at each predicted step, used both as the prediction
            targets and to overwrite boundary nodes during forecasting.
            Dims: same as the prediction.
        score_fn : Callable
            The configured scoring rule from ``neural_lam.metrics``, called
            as ``score_fn(prediction, target, pred_std, mask=...)``.
        interior_mask_bool : torch.Tensor
            Shape ``(num_grid_nodes,)``, boolean. ``True`` for interior
            nodes; passed as ``mask`` to ``score_fn`` so that only interior
            nodes are scored.
        per_var_std : torch.Tensor or None
            Shape ``(num_state_vars,)``. Constant per-variable standard
            deviation to score with when the forecaster does not predict its
            own std, otherwise ``None``.

        Returns
        -------
        batch_loss : torch.Tensor
            Scalar. The full training loss for the batch, to take gradients
            of.
        loss_components : dict of {str: torch.Tensor}
            Scalar loss-related quantities to log alongside the loss, keyed
            by component name. The wrapping module prefixes the names with
            the training phase. Empty when the objective has no separate
            components worth logging.
        """
