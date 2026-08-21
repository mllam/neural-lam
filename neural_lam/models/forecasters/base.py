"""Base class for forecasters."""

# Standard library
from abc import ABC, abstractmethod

# Third-party
import torch
from torch import nn

# Local
from ...datastore import BaseDatastore


class BaseForecaster(nn.Module, ABC):
    """
    Generic forecaster capable of mapping from a set of initial states,
    forcing and forces and previous states into a full forecast of the
    requested length.

    A forecaster owns its complete training objective
    (``compute_training_loss``) and produces forecasts through ``forward``.
    What ``forward`` returns differs between families of forecaster, so each
    family pins its own signature rather than this class fixing one:
    ``BaseDeterministicForecaster`` returns a single forecast (optionally
    with a predicted std), ``BaseEnsembleForecaster`` takes a member count
    and returns an ensemble. Every concrete forecaster therefore has exactly
    one entry point for producing forecasts. A forecaster that fits neither
    contract subclasses this class directly and pins a third, paired with a
    Lightning module that evaluates it.
    """

    boundary_mask: torch.Tensor
    interior_mask: torch.Tensor

    def __init__(self, datastore: BaseDatastore) -> None:
        """
        Initialize the forecaster.

        Parameters
        ----------
        datastore : BaseDatastore
            The datastore this forecaster is built for, providing grid
            metadata, boundary masks and standardization statistics.
        """
        super().__init__()
        self.datastore = datastore

        # Every forecaster on a limited-area grid overwrites boundary nodes
        # with their true values, so the masks live here rather than with
        # any one way of producing a forecast
        boundary_mask = (
            torch.tensor(datastore.boundary_mask.values, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        self.register_buffer("boundary_mask", boundary_mask, persistent=False)
        self.register_buffer(
            "interior_mask", 1.0 - self.boundary_mask, persistent=False
        )

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
    def compute_training_loss(
        self,
        init_states: torch.Tensor,
        forcing_features: torch.Tensor,
        target_states: torch.Tensor,
        interior_mask_bool: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute the training objective for one batch.

        The forecaster owns its complete training objective: which forecasts
        to produce from the batch, which loss terms to compute from them and
        how to combine those terms into a single scalar. The wrapping
        ``BaseForecastingModule`` only injects the interior mask, logs the
        returned components and optimizes the returned loss.

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
        interior_mask_bool : torch.Tensor
            Shape ``(num_grid_nodes,)``, boolean. ``True`` for interior
            nodes; passed as ``mask`` to the scoring rule so that only
            interior nodes are scored.

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
