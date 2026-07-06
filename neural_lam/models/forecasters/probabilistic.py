"""Forecasters producing probabilistic (ensemble) forecasts."""

# Standard library
from abc import abstractmethod
from typing import Callable

# Third-party
import torch

# Local
from ...datastore import BaseDatastore
from ..step_predictors.base import StepPredictor
from .autoregressive import ARForecaster
from .base import Forecaster


class ProbabilisticForecaster(Forecaster):
    """
    Forecaster whose forecasts are samples from a predictive distribution.

    Adds the capability that probabilistic evaluation and ensemble-based
    objectives build on: sampling an ensemble of forecasts. How the
    members are produced (auto-regressive sampling, diffusion, ...) is
    left to subclasses; consumers only rely on the shape of the returned
    ensemble.
    """

    @abstractmethod
    def sample_ensemble(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        boundary_states: torch.Tensor,
        num_members: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Sample an ensemble of forecasts.

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
        boundary_states : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``. True
            state values used only to overwrite boundary nodes at each
            predicted step, identically in every member. Dims: same as one
            member.
        num_members : int or None
            Number of ensemble members ``S`` to sample. When ``None``, the
            forecaster's configured ensemble size is used.

        Returns
        -------
        ensemble : torch.Tensor
            Shape ``(B, S, pred_steps, num_grid_nodes, num_state_vars)``.
            The sampled forecasts, stacked along the ensemble dimension
            ``S``.
        ensemble_std : torch.Tensor or None
            Shape ``(B, S, pred_steps, num_grid_nodes, num_state_vars)``
            when the forecaster predicts an std, otherwise ``None``. Dims:
            same as ``ensemble``.
        """


class ProbabilisticARForecaster(ARForecaster, ProbabilisticForecaster):
    """
    Auto-regressive forecaster for step predictors that sample their output.

    Each call to the wrapped predictor draws a fresh sample of the next
    state, so the inherited ``ARForecaster.forward`` unrolls one sampled
    trajectory. This class adds ensemble forecasting on top: unrolling
    several trajectories and stacking them along an ensemble dimension.
    The default training objective scores the ensemble mean using the
    scoring rule passed to ``compute_training_loss`` (from
    ``neural_lam.metrics``); forecasters with model-specific objectives
    (ensemble scoring rules, variational objectives) override
    ``compute_training_loss``.
    """

    def __init__(
        self,
        predictor: StepPredictor,
        datastore: BaseDatastore,
        ensemble_size: int,
    ) -> None:
        """
        Initialize the ProbabilisticARForecaster.

        Parameters
        ----------
        predictor : StepPredictor
            The predictor to use for each step. Each call should draw a
            fresh sample of the next state.
        datastore : BaseDatastore
            The datastore providing grid metadata and boundary masks.
        ensemble_size : int
            Number of ensemble members to sample when no explicit member
            count is given, in particular for the training objective.
        """
        super().__init__(predictor, datastore)
        if ensemble_size < 1:
            raise ValueError(
                f"ensemble_size must be at least 1, got {ensemble_size}"
            )
        self.ensemble_size = ensemble_size

    def sample_ensemble(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        boundary_states: torch.Tensor,
        num_members: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Sample an ensemble of forecasts.

        Unrolls ``num_members`` independent forecasts, each sampling fresh
        randomness at every step, and stacks them along a new ensemble
        dimension after the batch dimension.

        Parameters
        ----------
        init_states : torch.Tensor
            Shape ``(B, 2, num_grid_nodes, num_state_vars)``. The two initial
            states ``[X_{t-1}, X_t]`` used to start each rollout from. Dims:
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
        boundary_states : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``. True
            state values used only to overwrite boundary nodes at each AR
            step, identically in every member. Dims: same as one member.
        num_members : int or None
            Number of ensemble members ``S`` to sample. When ``None``,
            ``self.ensemble_size`` is used.

        Returns
        -------
        ensemble : torch.Tensor
            Shape ``(B, S, pred_steps, num_grid_nodes, num_state_vars)``.
            The sampled forecasts, stacked along the ensemble dimension
            ``S``.
        ensemble_std : torch.Tensor or None
            Shape ``(B, S, pred_steps, num_grid_nodes, num_state_vars)``
            when the wrapped predictor outputs an std, otherwise ``None``.
            Dims: same as ``ensemble``.
        """
        if num_members is None:
            num_members = self.ensemble_size

        member_list = []
        member_std_list = []
        for _ in range(num_members):
            prediction, pred_std = self(
                init_states, forcing_features, boundary_states
            )
            member_list.append(prediction)
            if pred_std is not None:
                member_std_list.append(pred_std)

        ensemble = torch.stack(member_list, dim=1)
        ensemble_std = (
            torch.stack(member_std_list, dim=1) if member_std_list else None
        )
        return ensemble, ensemble_std

    def compute_training_loss(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        target_states: torch.Tensor,
        score_metric: Callable[..., torch.Tensor],
        interior_mask_bool: torch.Tensor,
        per_var_std: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Score the ensemble mean with the given ``score_metric``.

        Samples an ensemble of ``self.ensemble_size`` forecasts, averages
        the members into an ensemble mean forecast, scores it against the
        target states on interior nodes and averages over batch and time.

        Parameters
        ----------
        init_states : torch.Tensor
            Shape ``(B, 2, num_grid_nodes, num_state_vars)``. The two initial
            states ``[X_{t-1}, X_t]`` used to start each rollout from. Dims:
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
            targets and to overwrite boundary nodes during the rollouts.
            Dims: same as one ensemble member.
        score_metric : Callable
            The configured scoring rule from ``neural_lam.metrics``, called
            as ``score_metric(prediction, target, pred_std, mask=...)``.
        interior_mask_bool : torch.Tensor
            Shape ``(num_grid_nodes,)``, boolean. ``True`` for interior
            nodes; passed as ``mask`` to ``score_metric`` so that only interior
            nodes are scored.
        per_var_std : torch.Tensor or None
            Shape ``(num_state_vars,)``. Constant per-variable standard
            deviation to score with when the wrapped predictor does not
            output an std, otherwise ``None``.

        Returns
        -------
        batch_loss : torch.Tensor
            Scalar. The scoring rule applied to the ensemble mean, averaged
            over batch and time.
        loss_components : dict of {str: torch.Tensor}
            Empty; this objective has no separate components.
        """
        ensemble, ensemble_std = self.sample_ensemble(
            init_states, forcing_features, target_states
        )
        ensemble_mean = ensemble.mean(dim=1)
        if ensemble_std is not None:
            pred_std = ensemble_std.mean(dim=1)
        else:
            pred_std = per_var_std

        batch_loss = torch.mean(
            score_metric(
                ensemble_mean,
                target_states,
                pred_std,
                mask=interior_mask_bool,
            )
        )
        return batch_loss, {}
