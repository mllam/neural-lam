"""Forecasters trained by scoring a single deterministic forecast."""

# Standard library
from typing import Optional

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

    ``compute_training_loss`` produces one forecast and scores it, and
    ``score`` applies the same metric to an already-produced forecast for
    reporting. ``forward`` is left abstract; see
    ``DeterministicARForecaster`` for the auto-regressive combination.
    """

    def __init__(
        self,
        datastore: BaseDatastore,
        config: NeuralLAMConfig | None = None,
        loss: str = "wmse",
    ) -> None:
        """
        Set up the scoring rule and the constant ``pred_std`` fallback.

        Parameters
        ----------
        datastore : BaseDatastore
            The datastore providing the state standardization statistics
            used to compute ``per_var_std``.
        config : NeuralLAMConfig or None, optional
            Configuration used to compute the constant per-variable std
            substituted for ``pred_std`` when the forecast carries no std of
            its own. Required in that case for ``score`` and
            ``compute_training_loss`` to work (they raise ``ValueError`` via
            ``_resolve_pred_std`` otherwise); forecasters used purely for
            inference can omit it. Default ``None``.
        loss : str, optional
            The scoring rule (from ``neural_lam.metrics``) applied by
            ``compute_training_loss``, stored as ``self.loss``. Default
            ``"wmse"``.
        """
        super().__init__(datastore=datastore)
        self.loss = metrics.get_metric(loss)

        # Whether the fallback is needed is settled per call in
        # _resolve_pred_std
        per_var_std = (
            get_per_var_std(config=config, datastore=datastore)
            if config is not None
            else None
        )
        self.register_buffer("per_var_std", per_var_std, persistent=False)

    def compute_training_loss(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        target_states: torch.Tensor,
        interior_mask_bool: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Score a single forecast with ``self.loss``.

        Produces one forecast over every predicted step, scores it against
        the target states on interior nodes and averages over batch and
        time. Callers that already hold a forecast should score that one
        with ``score`` rather than producing another here.

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
            is batch size, ``pred_steps`` is the number of predicted steps,
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
        prediction, pred_std = self(
            init_states, forcing_features, target_states
        )
        step_losses = self.score(
            prediction,
            target_states,
            pred_std,
            mask=interior_mask_bool,
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
            available (this forecaster was constructed without ``config``).
        """
        if pred_std is not None:
            return pred_std
        if self.per_var_std is None:
            raise ValueError(
                "No pred_std available for scoring: the forecast carries no "
                "std and this forecaster has no per_var_std fallback (it was "
                "constructed without config). Pass config to the "
                "constructor, or use a predictor that outputs its own std."
            )
        return self.per_var_std

    def score(
        self,
        prediction: torch.Tensor,
        target_states: torch.Tensor,
        pred_std: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        average_grid: bool = True,
        sum_vars: bool = True,
    ) -> torch.Tensor:
        """
        Apply this forecaster's scoring rule to an already-produced forecast.

        Resolves ``pred_std`` via ``_resolve_pred_std`` (substituting
        ``self.per_var_std`` when ``None``), then applies ``self.loss``. Used
        to report the loss on a forecast the caller already has, at a
        reduction of its choosing, without recomputing it.

        Only the loss goes through here. Metrics unrelated to the training
        objective depend on nothing but the shapes of the three tensors
        below, so callers compute those directly from
        ``neural_lam.metrics``.

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
        mask : torch.Tensor or None, optional
            Shape ``(num_grid_nodes,)``, boolean. Forwarded to ``self.loss``.
        average_grid : bool, optional
            Forwarded to ``self.loss``. Default ``True``.
        sum_vars : bool, optional
            Forwarded to ``self.loss``. Default ``True``.

        Returns
        -------
        torch.Tensor
            The scoring rule's output; shape depends on ``average_grid`` and
            ``sum_vars`` (see ``neural_lam.metrics``).

        Raises
        ------
        ValueError
            If ``pred_std`` is ``None`` and no ``per_var_std`` fallback is
            available; see ``_resolve_pred_std``.
        """
        pred_std = self._resolve_pred_std(pred_std)
        return self.loss(
            prediction,
            target_states,
            pred_std,
            mask=mask,
            average_grid=average_grid,
            sum_vars=sum_vars,
        )


class DeterministicARForecaster(ARForecaster, DeterministicForecaster):
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
        super().__init__(
            predictor=predictor,
            datastore=datastore,
            config=config,
            loss=loss,
        )
