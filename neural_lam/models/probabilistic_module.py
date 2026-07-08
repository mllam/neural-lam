"""Lightning module evaluating probabilistic forecasters as ensembles."""

# Standard library
import warnings

# Third-party
import torch

# Local
from .. import metrics
from .forecasters.probabilistic import ProbabilisticForecaster
from .module import ForecasterModule


class ProbabilisticForecasterModule(ForecasterModule):
    """
    Lightning module for forecasters that sample ensemble forecasts.

    Training is inherited unchanged from ``ForecasterModule``: the wrapped
    forecaster assembles its own training loss. Validation and testing are
    ensemble based instead of deterministic: an ensemble is sampled from
    the forecaster and scored through its ensemble mean (root-mean-squared
    error of the ensemble mean). The module only assumes that the
    forecaster can sample ensemble forecasts of the correct shape; it makes
    no assumption on how the members are produced.
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
            ``ForecasterModule.__init__`` (``forecaster``, ``config``,
            ``datastore``, ...).
        eval_ensemble_size : int
            Number of ensemble members sampled during validation and
            testing.
        **kwargs
            Keyword arguments forwarded to ``ForecasterModule.__init__``
            (``lr``, ...).
        """
        super().__init__(*args, **kwargs)
        if eval_ensemble_size < 1:
            raise ValueError(
                "eval_ensemble_size must be at least 1, "
                f"got {eval_ensemble_size}"
            )
        self.eval_ensemble_size = eval_ensemble_size
        self.val_metrics = {"ens_mse": []}
        self.test_metrics = {"ens_mse": []}

    def _ensemble_step(self, batch, phase: str):
        """
        Sample an ensemble and score its mean against the target states.

        Shared by ``validation_step`` and ``test_step``: samples
        ``self.eval_ensemble_size`` members, scores the ensemble mean with
        plain (unweighted) MSE on interior nodes, logs the root-mean-squared
        error per configured rollout step and averaged over the rollout
        under the given phase's prefix.

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

        time_step_mse = torch.mean(
            metrics.mse(
                ensemble_mean,
                target_states,
                std_placeholder,
                mask=self.interior_mask_bool,
            ),
            dim=0,
        )
        time_step_rmse = torch.sqrt(time_step_mse)
        mean_rmse = torch.mean(time_step_rmse)
        self._warn_skipped_val_steps(len(time_step_rmse), phase)

        log_dict = {
            f"{phase}_loss_unroll{step}": time_step_rmse[step - 1]
            for step in self.hparams.val_steps_to_log
            if step <= len(time_step_rmse)
        }
        log_dict[f"{phase}_mean_loss"] = mean_rmse
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )

        return metrics.mse(
            ensemble_mean,
            target_states,
            std_placeholder,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )

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

    def on_test_epoch_end(self):
        """
        Perform actions at the end of the test epoch.

        Aggregates ensemble test metrics. Overrides
        ``ForecasterModule.on_test_epoch_end``, which also handles spatial
        loss maps and example plots that ``test_step`` here does not
        populate.
        """
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
