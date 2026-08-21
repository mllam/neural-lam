"""Forecasters that sample their forecast from a predictive distribution."""

# Standard library

# Third-party
import torch

# Local
from ...datastore import BaseDatastore
from ..step_predictors.base import StepPredictor
from .autoregressive import unroll_forecast
from .base import BaseForecaster


class BaseProbabilisticForecaster(BaseForecaster):
    """
    Forecaster whose ``forward`` samples from a predictive distribution.

    ``forward`` keeps the signature every forecaster shares and returns one
    forecast, drawn afresh on each call. An ensemble is therefore not a
    different kind of output but repeated sampling, which is all
    ``sample_ensemble`` does. How a single sample is produced
    (auto-regressive sampling, diffusion, ...) is left to subclasses.

    When ``forward`` returns a ``pred_std``, it is that one sample's own
    predicted std, not a std describing the spread across members. Stacked
    by ``sample_ensemble``, the predictive distribution is then a mixture of
    ``S`` Gaussians, one per member: ``p(x) = mean_s N(x; ensemble[:, s],
    per_member_std[:, s]**2)``, not a single Gaussian. In particular, the
    variance of that mixture is not the average of the per-member variances:
    it also includes the spread between the member means.
    """

    def sample_ensemble(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        boundary_states: torch.Tensor,
        num_members: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Sample an ensemble of forecasts.

        Calls ``forward`` ``num_members`` times, each drawing fresh
        randomness, and stacks the results along a new ensemble dimension
        after the batch dimension.

        Members are drawn sequentially, one full forecast at a time, so cost
        grows linearly with ``num_members``. They are independent given the
        inputs, so this is only an implementation choice: a subclass whose
        sampling batches cheaply can override this to fold the member
        dimension into the batch dimension, at proportionally higher peak
        memory.

        Parameters
        ----------
        init_states : torch.Tensor
            Shape ``(B, 2, num_grid_nodes, num_state_vars)``. The two initial
            states ``[X_{t-1}, X_t]`` used to start each member from. Dims:
            ``B`` is batch size, ``2`` is the time index (``[X_{t-1}, X_t]``),
            ``num_grid_nodes`` is the number of spatial nodes, and
            ``num_state_vars`` is the state feature dimension.
        forcing_features : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_forcing_vars)``.
            External forcings provided at each predicted step. Dims: ``B``
            is batch size, ``pred_steps`` is the forecast length,
            ``num_grid_nodes`` is the number of spatial nodes, and
            ``num_forcing_vars`` is the forcing feature dimension (already
            concatenated past/current/future windows).
        boundary_states : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``. True
            state values used only to overwrite boundary nodes at each
            predicted step, identically in every member. Dims: same as one
            member.
        num_members : int
            Number of ensemble members ``S`` to sample.

        Returns
        -------
        ensemble : torch.Tensor
            Shape ``(B, S, pred_steps, num_grid_nodes, num_state_vars)``.
            The sampled forecasts, stacked along the ensemble dimension
            ``S``.
        per_member_std : torch.Tensor or None
            Shape ``(B, S, pred_steps, num_grid_nodes, num_state_vars)``.
            Each member's own predicted std (see the class docstring for
            why the ensemble is then a mixture, not this averaged with the
            others), when the forecaster predicts an std, otherwise
            ``None``. Dims: same as ``ensemble``.

        Raises
        ------
        ValueError
            If ``num_members`` is less than 1.
        """
        if num_members < 1:
            raise ValueError(
                f"num_members must be at least 1, got {num_members}"
            )

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
        # After stacking, ensemble has shape
        # (B, S, pred_steps, num_grid_nodes, num_state_vars)
        per_member_std = (
            torch.stack(member_std_list, dim=1) if member_std_list else None
        )
        return ensemble, per_member_std


class BaseProbabilisticARForecaster(BaseProbabilisticForecaster):
    """
    Probabilistic forecaster sampling each forecast auto-regressively.

    Each call to the wrapped predictor draws a fresh sample of the next
    state, so one unrolling is one member and ``sample_ensemble`` gets
    independent trajectories for free.

    It supplies no training objective, and so remains abstract in
    ``compute_training_loss``. There is no default that fits every
    stochastic model: scoring the ensemble mean with a pointwise metric
    only rewards the mean being right, giving the model no incentive to
    keep a calibrated spread, and risks training it to collapse the
    ensemble to a point estimate. Concrete subclasses define an objective
    appropriate to how they are meant to be trained (e.g. an ensemble
    scoring rule such as CRPS, or a variational objective), along with
    whatever configuration that objective needs.
    """

    def __init__(
        self,
        predictor: StepPredictor,
        datastore: BaseDatastore,
    ) -> None:
        """
        Initialize the BaseProbabilisticARForecaster.

        Parameters
        ----------
        predictor : StepPredictor
            The predictor to use for each AR step. Samples its output, so
            that each rollout is an independent member.
        datastore : BaseDatastore
            The datastore providing grid metadata and boundary masks.
        """
        super().__init__(datastore=datastore)
        self.predictor = predictor

    @property
    def predicts_std(self) -> bool:
        """
        Whether the forecaster predicts standard deviation.

        Returns
        -------
        bool
            ``True`` if the wrapped predictor predicts standard deviation,
            ``False`` otherwise.
        """
        return self.predictor.predicts_std

    def forward(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        boundary_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Unroll one sampled auto-regressive forecast.

        Parameters
        ----------
        init_states : torch.Tensor
            Shape ``(B, 2, num_grid_nodes, num_state_vars)``. The two initial
            states ``[X_{t-1}, X_t]`` used to start the rollout from.
        forcing_features : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_forcing_vars)``.
            Forcing features for each predicted step; ``pred_steps`` defines
            the rollout length.
        boundary_states : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``. True
            state values used ONLY to overwrite boundary nodes at each AR
            step.

        Returns
        -------
        prediction : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``. One
            sampled trajectory. Dims: same as ``boundary_states``.
        pred_std : torch.Tensor or None
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)`` when
            the wrapped predictor outputs an std, otherwise ``None``. Dims:
            same as ``prediction``.
        """
        return unroll_forecast(
            self.predictor,
            init_states,
            forcing_features,
            boundary_states,
            self.boundary_mask,
            self.interior_mask,
        )
