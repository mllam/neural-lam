"""Forecaster that uses an auto-regressive strategy to unroll a forecast."""

# Third-party
import torch

# Local
from ... import metrics
from ...config import NeuralLAMConfig
from ...datastore import BaseDatastore
from ...loss_weighting import get_state_feature_weighting
from ..step_predictors.base import StepPredictor
from .base import Forecaster


class ARForecaster(Forecaster):
    """
    Subclass of Forecaster that uses an auto-regressive strategy to
    unroll a forecast. Makes use of a StepPredictor at each AR step.
    """

    def __init__(
        self,
        predictor: StepPredictor,
        datastore: BaseDatastore,
        config: NeuralLAMConfig | None = None,
        loss: str = "wmse",
    ) -> None:
        """
        Initialize the ARForecaster.

        Parameters
        ----------
        predictor : StepPredictor
            The predictor to use for each step.
        datastore : BaseDatastore
            The datastore providing grid metadata and boundary masks.
        config : NeuralLAMConfig or None
            Configuration used to compute the constant per-variable std
            substituted for ``pred_std`` when ``predictor`` does not output
            its own (see ``per_var_std``). Only required for that case;
            forecasters used purely for inference can omit it.
        loss : str, default "wmse"
            The scoring rule (from ``neural_lam.metrics``) used by
            ``compute_training_loss`` and stored as ``self.loss``.
        """
        super().__init__()
        self.predictor = predictor

        # Register boundary/interior masks on the forecaster, not the predictor
        boundary_mask = (
            torch.tensor(datastore.boundary_mask.values, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        self.register_buffer("boundary_mask", boundary_mask, persistent=False)
        self.register_buffer(
            "interior_mask", 1.0 - self.boundary_mask, persistent=False
        )

        self.loss = metrics.get_metric(loss)

        # Store per_var_std here if the predictor does not output its own std
        if not self.predicts_std and config is not None:
            da_state_stats = datastore.get_standardization_dataarray(
                category="state"
            )
            state_feature_weights = get_state_feature_weighting(
                config=config, datastore=datastore
            )
            diff_std = torch.tensor(
                da_state_stats.state_diff_std_standardized.values,
                dtype=torch.float32,
            )
            feature_weights_t = torch.tensor(
                state_feature_weights, dtype=torch.float32
            )
            self.register_buffer(
                "per_var_std",
                diff_std / torch.sqrt(feature_weights_t),
                persistent=False,
            )
        else:
            self.per_var_std = None

    @property
    def predicts_std(self) -> bool:
        """
        Whether the forecaster predicts standard deviation.

        Returns
        -------
        bool
            ``True`` if the forecaster predicts standard deviation,
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
        Unroll the autoregressive model: at each step ``i`` call
        ``self.predictor`` to produce the next state, then overwrite boundary
        nodes with the true value from ``boundary_states[:, i]``.

        Parameters
        ----------
        init_states : torch.Tensor
            Shape ``(B, 2, num_grid_nodes, num_state_vars)``. The two initial
            states ``[X_{t-1}, X_t]`` used to start the rollout from. Dims:
            ``B`` is batch size, ``2`` initial time steps (``X_{t-1}, X_t``),
            ``num_grid_nodes`` is the number of spatial nodes, and
            ``num_state_vars`` is the number of state variables.
        forcing_features : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_forcing_vars)``.
            Forcing features for each predicted step; ``pred_steps`` defines
            the rollout length. Dims: ``B`` is batch size, ``pred_steps`` is
            the number of predicted steps, ``num_grid_nodes`` is the
            number of spatial nodes, and ``num_forcing_vars`` is the number
            of forcing variables (already concatenated past/current/future
            windows).
        boundary_states : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``.
            True state values used ONLY to overwrite boundary nodes at
            each AR step. The interior prediction at step ``i`` must not
            depend on ``boundary_states[:, i]`` in any other way. Dims:
            ``B`` is batch size, ``pred_steps`` is the number of
            predicted steps, ``num_grid_nodes`` is the number of spatial
            nodes, and
            ``num_state_vars`` is the state feature dimension.

        Returns
        -------
        prediction : torch.Tensor
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)``. Stacked
            per-step forecasts (with boundary overwritten by the true
            value). Dims: same as ``boundary_states``.
        pred_std : torch.Tensor or None
            Shape ``(B, pred_steps, num_grid_nodes, num_state_vars)`` when the
            wrapped predictor outputs an std, otherwise ``None`` (in which
            case ``ForecasterModule`` substitutes the constant
            per-variable std). Dims: same as ``prediction``.
        """

        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []
        pred_steps = forcing_features.shape[1]

        for i in range(pred_steps):
            forcing = forcing_features[:, i]
            boundary_state = boundary_states[:, i]

            pred_state, pred_std = self.predictor(
                prev_state, prev_prev_state, forcing
            )

            # Overwrite boundary with true state using ARForecaster's mask
            new_state = (
                self.boundary_mask * boundary_state
                + self.interior_mask * pred_state
            )

            prediction_list.append(new_state)
            if pred_std is not None:
                pred_std_list.append(pred_std)

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = new_state

        prediction = torch.stack(prediction_list, dim=1)
        # If predictor outputs std, stack it; otherwise return None so
        # callers can substitute the constant per_var_std
        if pred_std_list:
            pred_std = torch.stack(pred_std_list, dim=1)
        else:
            pred_std = None

        return prediction, pred_std

    def compute_training_loss(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        target_states: torch.Tensor,
        interior_mask_bool: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Score the deterministic rollout with ``self.loss``.

        Unrolls a single forecast over the full rollout, scores it against
        the target states on interior nodes and averages over batch and
        time.

        Parameters
        ----------
        init_states : torch.Tensor
            Shape ``(B, 2, num_grid_nodes, num_state_vars)``. The two initial
            states ``[X_{t-1}, X_t]`` used to start the rollout from. Dims:
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
            targets and to overwrite boundary nodes during the rollout.
            Dims: same as the prediction.
        interior_mask_bool : torch.Tensor
            Shape ``(num_grid_nodes,)``, boolean. ``True`` for interior
            nodes; passed as ``mask`` to ``self.loss`` so that only interior
            nodes are scored.

        Returns
        -------
        batch_loss : torch.Tensor
            Scalar. The scoring rule applied to the rollout, averaged over
            batch and time.
        loss_components : dict of {str: torch.Tensor}
            Empty; the deterministic objective has no separate components.
        """
        prediction, pred_std = self(
            init_states, forcing_features, target_states
        )
        if pred_std is None:
            pred_std = self.per_var_std

        batch_loss = torch.mean(
            self.loss(
                prediction,
                target_states,
                pred_std,
                mask=interior_mask_bool,
            )
        )
        return batch_loss, {}
