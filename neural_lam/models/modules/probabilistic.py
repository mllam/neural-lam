"""Lightning module evaluating probabilistic forecasters as ensembles."""

# Standard library
import warnings

# Third-party
import torch

# Local
from ... import metrics
from ..forecasters.probabilistic import ProbabilisticForecaster
from .base import BaseForecasterModule


class ProbabilisticForecasterModule(BaseForecasterModule):
    """
    Lightning module for forecasters that sample ensemble forecasts.

    Training is inherited unchanged from ``BaseForecasterModule``: the
    wrapped forecaster assembles its own training loss. Validation and
    testing are ensemble based instead of deterministic: an ensemble is
    sampled from the forecaster and scored through its ensemble mean
    (root-mean-squared error of the ensemble mean). The module only assumes
    that the forecaster can sample ensemble forecasts of the correct shape;
    it makes no assumption on how the members are produced.
    """

    # The wrapped forecaster must be able to sample ensemble forecasts
    forecaster: ProbabilisticForecaster

    def __init__(self, *args, eval_ensemble_size: int, **kwargs):
        """
        Initialize the module and store the evaluation ensemble size.

        Parameters
        ----------
        *args
            Positional arguments forwarded to
            ``BaseForecasterModule.__init__`` (``forecaster``, ``config``,
            ``datastore``, ...).
        eval_ensemble_size : int
            Number of ensemble members sampled during validation and
            testing.
        **kwargs
            Keyword arguments forwarded to ``BaseForecasterModule.__init__``
            (``lr``, ...).
        """
        super().__init__(*args, **kwargs)
        if eval_ensemble_size < 1:
            raise ValueError(
                "eval_ensemble_size must be at least 1, "
                f"got {eval_ensemble_size}"
            )
        self.eval_ensemble_size = eval_ensemble_size
        self.val_metrics: dict[str, list] = {"ens_mse": []}
        self.test_metrics: dict[str, list] = {"ens_mse": []}

    def _ensemble_step(self, batch, phase: str):
        """
        Sample an ensemble and compute the per-variable MSE of its mean.

        Shared by ``validation_step`` and ``test_step``: samples
        ``self.eval_ensemble_size`` members and scores the ensemble mean
        with plain (unweighted) MSE on interior nodes. Only the squared
        errors are computed here; they are reduced to an RMSE once per
        epoch by ``_log_ensemble_rmse``, since the square root has to be
        taken after averaging over every sample rather than per batch.

        This is a diagnostic metric, not the training loss: it always
        scores the ensemble mean with plain MSE, regardless of what
        objective ``compute_training_loss`` actually trains on.

        Parameters
        ----------
        batch : tuple
            The batch of data.
        phase : str
            Logging phase, either ``"val"`` or ``"test"``.

        Returns
        -------
        torch.Tensor
            Per-variable ensemble-mean MSE, shape
            ``(B, pred_steps, num_state_vars)``, for epoch-end aggregation.
        """
        init_states, target_states, forcing_features, _ = batch
        ensemble, _ = self.forecaster.sample_ensemble(
            init_states,
            forcing_features,
            target_states,
            num_members=self.eval_ensemble_size,
        )
        ensemble_mean = ensemble.mean(dim=1)
        # metrics.mse ignores the std argument, but requires one
        std_placeholder = torch.ones(
            target_states.shape[-1], device=target_states.device
        )

        entry_mses = metrics.mse(
            ensemble_mean,
            target_states,
            std_placeholder,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, num_state_vars)
        self._warn_skipped_steps(entry_mses.shape[1], phase)

        return entry_mses

    def _log_ensemble_rmse(self, entry_mse_list, phase: str) -> None:
        """
        Log the ensemble-mean RMSE accumulated over a full epoch.

        Averages the per-variable MSEs collected by ``_ensemble_step`` over
        every sample of the epoch (across both batches and devices) and sums
        them over variables, and only then takes the square root. Rooting
        each batch's MSE and averaging those roots instead would report a
        different quantity, since the square root does not commute with the
        averaging.

        Parameters
        ----------
        entry_mse_list : list of torch.Tensor
            Per-batch per-variable ensemble-mean MSEs, each of shape
            ``(B, pred_steps, num_state_vars)``, as returned by
            ``_ensemble_step``.
        phase : str
            Logging phase, either ``"val"`` or ``"test"``.
        """
        # Collective: must be reached by every rank, so it precedes the
        # rank-zero check below.
        entry_mses = self.all_gather_cat(torch.cat(entry_mse_list, dim=0))
        # (total_samples, pred_steps, num_state_vars)

        if not self.trainer.is_global_zero:
            return

        time_step_rmse = torch.sqrt(entry_mses.sum(dim=-1).mean(dim=0))
        # (pred_steps,)

        log_dict = {
            f"{phase}_ens_rmse_unroll{step}": time_step_rmse[step - 1]
            for step in self.hparams.val_steps_to_log
            if step <= len(time_step_rmse)
        }
        log_dict[f"{phase}_mean_ens_rmse"] = torch.mean(time_step_rmse)
        # No sync_dist: the values are already gathered above and only rank
        # zero reaches this point, so a collective here would hang DDP.
        self.log_dict(log_dict, rank_zero_only=True)

    def validation_step(self, batch, batch_idx):
        """
        Perform a single ensemble validation step.

        Scores the ensemble mean against the target states (see
        ``_ensemble_step``) and collects per-variable ensemble-mean MSE for
        epoch-end aggregation.

        Parameters
        ----------
        batch : tuple
            The batch of data.
        batch_idx : int
            The index of the batch.
        """
        entry_mses = self._ensemble_step(batch, "val")
        self.val_metrics["ens_mse"].append(entry_mses)

    def test_step(self, batch, batch_idx):
        """
        Perform a single ensemble test step.

        Scores the ensemble mean against the target states (see
        ``_ensemble_step``) and collects per-variable ensemble-mean MSE for
        epoch-end aggregation.

        Parameters
        ----------
        batch : tuple
            The batch of data.
        batch_idx : int
            The index of the batch.
        """
        entry_mses = self._ensemble_step(batch, "test")
        self.test_metrics["ens_mse"].append(entry_mses)

    def on_validation_epoch_end(self):
        """
        Perform actions at the end of the validation epoch.

        Logs the epoch's ensemble-mean RMSE, then defers to
        ``BaseForecasterModule.on_validation_epoch_end``, which aggregates
        the same per-variable MSEs into heatmaps and clears them.
        """
        self._log_ensemble_rmse(self.val_metrics["ens_mse"], "val")
        super().on_validation_epoch_end()

    def on_test_epoch_end(self):
        """
        Perform actions at the end of the test epoch.

        Logs the epoch's ensemble-mean RMSE and aggregates ensemble test
        metrics. Implements ``BaseForecasterModule.on_test_epoch_end``
        without the spatial loss maps and example plots that
        ``DeterministicForecasterModule`` adds, since ``test_step`` here
        does not populate them.
        """
        self._log_ensemble_rmse(self.test_metrics["ens_mse"], "test")
        self.aggregate_and_plot_metrics(self.test_metrics, prefix="test")

        if self.trainer.is_global_zero and self.hparams.metrics_watch:
            unmatched = set(self.hparams.metrics_watch) - self.matched_metrics
            if unmatched:
                warnings.warn(
                    "The following metrics in --metrics_watch "
                    "were not found during test phase: "
                    f"{sorted(unmatched)}. Ensure the metric prefix "
                    "matches the evaluation mode (expected 'test_')."
                )

        self.matched_metrics = set()

        for metric_list in self.test_metrics.values():
            metric_list.clear()
