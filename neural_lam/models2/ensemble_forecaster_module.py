# Third-party
import torch

# Local
from ..metrics import crps_loss
from .forecaster_module import ForecasterModule


class EnsembleForecasterModule(ForecasterModule):
    """
    Probabilistic grading module using CRPS for ensemble forecasts.

    Tensor shapes used by this module:
    - `init_states`: (B, 2, N, F)
    - `forcing_features`: (B, T, N, F_forcing)
    - `target_states`: (B, T, N, F)
    - `prediction`: (B, S, T, N, F)
      where B=batch size, S=ensemble members, T=lead time, N=nodes,
      F=state features.
    """

    def __init__(self, forecaster, args, config, datastore):
        """
        Initialize the ensemble forecasting module.

        Parameters
        ----------
        forecaster : ARForecastSampler
            Forecaster returning ensemble predictions with shape
            (B, S, T, N, F).
        args : Any
            Runtime args namespace.
        config : Any
            Experiment/model config object.
        datastore : BaseDatastore
            Datastore used for metadata and plotting in parent module.
        """
        super().__init__(forecaster, args, config, datastore)
        self.loss = crps_loss

    @staticmethod
    def _mask_node_dimension(tensor: torch.Tensor, interior_mask: torch.Tensor):
        """
        Apply interior-node mask to tensors with node dimension at index -2.

        Supported shapes:
        - (B, T, N, F)
        - (B, S, T, N, F)
        """
        if interior_mask.dim() == 2:
            mask_bool = interior_mask[:, 0].to(torch.bool)
        else:
            mask_bool = interior_mask.to(torch.bool)

        if tensor.dim() == 4:
            return tensor[:, :, mask_bool, :]
        if tensor.dim() == 5:
            return tensor[:, :, :, mask_bool, :]
        raise ValueError("Expected tensor with dim 4 or 5 for masking")

    def training_step(self, batch, batch_idx):
        """
        Train with CRPS over ensemble predictions.

        Uses parent `common_step`:
        - prediction: (B, S, T, N, F)
        - target_states: (B, T, N, F)
        """
        _ = batch_idx
        prediction, target_states, _, _ = self.common_step(batch)
        interior_mask = self.forecaster.step_predictor.interior_mask

        crps_entries = self.loss(
            prediction,
            target_states,
            mask=interior_mask,
            average_grid=False,
            sum_vars=False,
        )  # (B, T, N_int, F)
        train_loss = torch.mean(crps_entries)

        self.log(
            "train_loss",
            train_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        """
        Validate with CRPS and spread-error diagnostics.

        Logs:
        - `val_crps`: mean CRPS over batch/time/node/feature.
        - `val_spread`: mean ensemble variance.
        - `val_error`: mean squared error of ensemble mean against truth.
        """
        _ = batch_idx
        prediction, target_states, _, _ = self.common_step(batch)
        interior_mask = self.forecaster.step_predictor.interior_mask

        crps_entries = self.loss(
            prediction,
            target_states,
            mask=interior_mask,
            average_grid=False,
            sum_vars=False,
        )  # (B, T, N_int, F)
        crps_per_timestep = torch.mean(crps_entries, dim=(0, 2, 3))  # (T,)
        val_crps = torch.mean(crps_per_timestep)

        ensemble_mean = torch.mean(prediction, dim=1)  # (B, T, N, F)
        error_entries = (ensemble_mean - target_states) ** 2
        spread_entries = torch.var(prediction, dim=1, unbiased=False)

        error_entries = self._mask_node_dimension(error_entries, interior_mask)
        spread_entries = self._mask_node_dimension(spread_entries, interior_mask)
        val_error = torch.mean(error_entries)
        val_spread = torch.mean(spread_entries)

        self.log_dict(
            {
                "val_crps": val_crps,
                "val_spread": val_spread,
                "val_error": val_error,
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
