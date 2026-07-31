"""Forecasters trained by scoring a single deterministic forecast."""

# Standard library
from typing import Callable, Optional

# Third-party
import torch

# Local
from ... import metrics
from ...config import NeuralLAMConfig
from ...datastore import BaseDatastore
from ...loss_weighting import get_per_var_std
from ..step_predictors.base import StepPredictor
from .autoregressive import ARForecaster
from .base import Forecaster


class DeterministicForecaster(Forecaster):
    """
    Forecaster whose training objective is a scoring rule applied to a
    single forecast.

    Supplies the objective half of a forecaster: ``compute_training_loss``
    produces one forecast and scores it, and ``score`` applies a metric to
    an already-produced forecast for reporting. Neither makes any assumption
    about *how* that forecast is produced, so this composes with any way of
    implementing ``forward`` (see ``DeterministicARForecaster`` for the
    auto-regressive combination).

    Concrete subclasses must call ``_configure_scoring`` from their
    ``__init__`` to set up the scoring rule and the ``pred_std`` fallback.
    """

    def _configure_scoring(
        self,
        datastore: BaseDatastore,
        config: NeuralLAMConfig | None,
        loss: str,
    ) -> None:
        """
        Set up the scoring rule and the constant ``pred_std`` fallback.

        Called by concrete subclasses from ``__init__``, after
        ``torch.nn.Module`` initialization (buffers are registered here).

        Parameters
        ----------
        datastore : BaseDatastore
            The datastore providing the state standardization statistics
            used to compute ``per_var_std``.
        config : NeuralLAMConfig or None
            Configuration used to compute the constant per-variable std
            substituted for ``pred_std`` when the forecast carries no std of
            its own. Required in that case for ``score`` and
            ``compute_training_loss`` to work (they raise ``ValueError`` via
            ``_resolve_pred_std`` otherwise); forecasters used purely for
            inference can omit it.
        loss : str
            The scoring rule (from ``neural_lam.metrics``) applied by
            ``compute_training_loss``, stored as ``self.loss``.
        """
        self.loss = metrics.get_metric(loss)

        # Store per_var_std only if the forecast carries no std of its own
        if not self.predicts_std and config is not None:
            self.register_buffer(
                "per_var_std",
                get_per_var_std(config=config, datastore=datastore),
                persistent=False,
            )
        else:
            self.per_var_std = None

    def compute_step_losses(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        target_states: torch.Tensor,
        interior_mask_bool: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score a single forecast with ``self.loss``, per predicted step.

        This objective is a per-step scoring rule averaged over the rollout,
        so it decomposes into the contribution of each predicted step, which
        callers can report individually. ``compute_training_loss`` is the
        mean of what this returns.

        Parameters
        ----------
        init_states : torch.Tensor
            Shape ``(B, 2, num_grid_nodes, num_state_vars)``. The two initial
            states ``[X_{t-1}, X_t]`` used to start the forecast from.
        forcing_features : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_forcing_vars)``.
            External forcings provided at each predicted step.
        target_states : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``. True
            states at each predicted step, used both as the prediction
            targets and to overwrite boundary nodes while forecasting.
        interior_mask_bool : torch.Tensor
            Shape ``(num_grid_nodes,)``, boolean. ``True`` for interior
            nodes; passed as ``mask`` to ``self.loss`` so that only interior
            nodes are scored.

        Returns
        -------
        torch.Tensor
            Shape ``(pred_steps,)``. The scoring rule at each predicted
            step, averaged over the batch.

        Raises
        ------
        ValueError
            If the forecast carries no std of its own and no
            ``per_var_std`` fallback is available; see
            ``_resolve_pred_std``.
        """
        prediction, pred_std = self(
            init_states, forcing_features, target_states
        )
        pred_std = self._resolve_pred_std(pred_std)

        return torch.mean(
            self.loss(
                prediction,
                target_states,
                pred_std,
                mask=interior_mask_bool,
            ),
            dim=0,
        )

    def compute_training_loss(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        target_states: torch.Tensor,
        interior_mask_bool: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Score a single forecast with ``self.loss``.

        Produces one forecast over the full rollout, scores it against the
        target states on interior nodes and averages over batch and time.
        Callers wanting the per-step breakdown of this same objective
        should use ``compute_step_losses`` instead, which this averages.

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
            targets and to overwrite boundary nodes while forecasting.
            Dims: same as the prediction.
        interior_mask_bool : torch.Tensor
            Shape ``(num_grid_nodes,)``, boolean. ``True`` for interior
            nodes; passed as ``mask`` to ``self.loss`` so that only interior
            nodes are scored.

        Returns
        -------
        batch_loss : torch.Tensor
            Scalar. The scoring rule applied to the forecast, averaged over
            batch and time.
        loss_components : dict of {str: torch.Tensor}
            Empty; the deterministic objective has no separate components.

        Raises
        ------
        ValueError
            If the forecast carries no std of its own and no
            ``per_var_std`` fallback is available; see
            ``_resolve_pred_std``.
        """
        step_losses = self.compute_step_losses(
            init_states,
            forcing_features,
            target_states,
            interior_mask_bool,
        )
        return torch.mean(step_losses), {}

    def _resolve_pred_std(
        self, pred_std: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Return ``pred_std``, or the constant ``per_var_std`` fallback.

        Parameters
        ----------
        pred_std : torch.Tensor or None
            Predicted standard deviation as returned by ``forward``,
            possibly ``None``.

        Returns
        -------
        torch.Tensor
            ``pred_std`` unchanged when given; otherwise ``self.per_var_std``.

        Raises
        ------
        ValueError
            If ``pred_std`` is ``None`` and no ``per_var_std`` fallback is
            available (``predicts_std`` is False and this forecaster was
            constructed without ``config``).
        """
        if pred_std is not None:
            return pred_std
        if self.per_var_std is None:
            raise ValueError(
                "No pred_std available for scoring: predictor.predicts_std "
                "is False and this forecaster has no per_var_std fallback "
                "(it was constructed without config). Pass config to the "
                "constructor, or use a predictor that outputs its own std."
            )
        return self.per_var_std

    def score(
        self,
        prediction: torch.Tensor,
        target_states: torch.Tensor,
        pred_std: Optional[torch.Tensor],
        metric: Optional[Callable[..., torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        average_grid: bool = True,
        sum_vars: bool = True,
    ) -> torch.Tensor:
        """
        Score an already-produced prediction for reporting (not training).

        Resolves ``pred_std`` via ``_resolve_pred_std`` (substituting
        ``self.per_var_std`` when ``None``), then applies ``metric``
        (defaulting to ``self.loss``, the configured scoring rule).

        Parameters
        ----------
        prediction : torch.Tensor
            Shape ``(..., num_grid_nodes, num_state_vars)``. Forecast to
            score.
        target_states : torch.Tensor
            Shape ``(..., num_grid_nodes, num_state_vars)``. True states to
            score against. Dims: same as ``prediction``.
        pred_std : torch.Tensor or None
            Shape ``(..., num_grid_nodes, num_state_vars)``, or ``None``.
            Predicted standard deviation for ``prediction``; ``None`` when
            the forecast carries no std, in which case ``self.per_var_std``
            is substituted (see ``_resolve_pred_std`` for when this raises
            instead).
        metric : callable or None, optional
            Scoring function with the ``neural_lam.metrics`` signature
            ``(pred, target, pred_std, mask=None, average_grid=True,
            sum_vars=True) -> torch.Tensor``. Defaults to ``self.loss``.
        mask : torch.Tensor or None, optional
            Shape ``(num_grid_nodes,)``, boolean. Forwarded to ``metric``.
        average_grid : bool, optional
            Forwarded to ``metric``.
        sum_vars : bool, optional
            Forwarded to ``metric``.

        Returns
        -------
        torch.Tensor
            The metric's output; shape depends on ``average_grid`` and
            ``sum_vars`` (see ``neural_lam.metrics``).

        Raises
        ------
        ValueError
            If ``pred_std`` is ``None`` and no ``per_var_std`` fallback is
            available; see ``_resolve_pred_std``.
        """
        pred_std = self._resolve_pred_std(pred_std)
        metric_fn = self.loss if metric is None else metric
        return metric_fn(
            prediction,
            target_states,
            pred_std,
            mask=mask,
            average_grid=average_grid,
            sum_vars=sum_vars,
        )


class DeterministicARForecaster(DeterministicForecaster, ARForecaster):
    """
    Auto-regressive forecaster trained by scoring its single rollout.

    Combines the two orthogonal halves: ``ARForecaster`` supplies the
    auto-regressive ``forward``, ``DeterministicForecaster`` supplies the
    single-forecast training objective and the reporting ``score``.
    """

    def __init__(
        self,
        predictor: StepPredictor,
        datastore: BaseDatastore,
        config: NeuralLAMConfig | None = None,
        loss: str = "wmse",
    ) -> None:
        """
        Initialize the DeterministicARForecaster.

        Parameters
        ----------
        predictor : StepPredictor
            The predictor to use for each AR step.
        datastore : BaseDatastore
            The datastore providing grid metadata and boundary masks.
        config : NeuralLAMConfig or None
            Configuration used to compute the constant per-variable std
            substituted for ``pred_std`` when ``predictor`` does not output
            its own. Required in that case for ``score`` and
            ``compute_training_loss`` to work (they raise ``ValueError``
            otherwise); forecasters used purely for inference (``forward``)
            can omit it.
        loss : str, default "wmse"
            The scoring rule (from ``neural_lam.metrics``) applied by
            ``compute_training_loss``.
        """
        # DeterministicForecaster defines no __init__, so this initializes
        # the AR half (and torch.nn.Module) before any buffer is registered
        super().__init__(predictor, datastore)
        self._configure_scoring(datastore, config=config, loss=loss)
