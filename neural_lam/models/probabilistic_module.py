"""Lightning module evaluating probabilistic forecasters as ensembles."""

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
    forecaster assembles its own training loss. Validation is ensemble
    based instead of deterministic: an ensemble is sampled from the
    forecaster and scored through its ensemble mean (root-mean-squared
    error of the ensemble mean). The module only assumes that the
    forecaster can sample ensemble forecasts of the correct shape; it makes
    no assumption on how the members are produced.
    """

    # The wrapped forecaster must be able to sample ensemble forecasts
    forecaster: ProbabilisticForecaster

    def __init__(self, *args, eval_ensemble_size: int | None = None, **kwargs):
        """
        Initialize the module and store the evaluation ensemble size.

        Parameters
        ----------
        *args
            Positional arguments forwarded to
            ``ForecasterModule.__init__`` (``forecaster``, ``config``,
            ``datastore``, ...).
        eval_ensemble_size : int or None
            Number of ensemble members sampled during validation. ``None``
            uses the forecaster's configured ensemble size.
        **kwargs
            Keyword arguments forwarded to ``ForecasterModule.__init__``
            (``loss``, ``lr``, ...).
        """
        super().__init__(*args, **kwargs)
        if eval_ensemble_size is not None and eval_ensemble_size < 1:
            raise ValueError(
                "eval_ensemble_size must be at least 1, "
                f"got {eval_ensemble_size}"
            )
        self.eval_ensemble_size = eval_ensemble_size
        self.val_metrics = {"ens_mse": []}

    def validation_step(self, batch, batch_idx):
        """
        Perform a single ensemble validation step.

        Samples an ensemble from the forecaster and scores its ensemble
        mean against the target states on interior nodes. Logs the
        root-mean-squared error of the ensemble mean per configured rollout
        step and averaged over the rollout, and collects per-variable
        ensemble-mean MSE for epoch-end aggregation.

        Parameters
        ----------
        batch : tuple
            The batch of data.
        batch_idx : int
            The index of the batch.
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
        self._warn_skipped_val_steps(len(time_step_rmse), "val")

        val_log_dict = {
            f"val_loss_unroll{step}": time_step_rmse[step - 1]
            for step in self.hparams.val_steps_to_log
            if step <= len(time_step_rmse)
        }
        val_log_dict["val_mean_loss"] = mean_rmse
        self.log_dict(
            val_log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )

        entry_mses = metrics.mse(
            ensemble_mean,
            target_states,
            std_placeholder,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )
        self.val_metrics["ens_mse"].append(entry_mses)

    def test_step(self, batch, batch_idx):
        """
        Not supported: ensemble test evaluation is not implemented.

        Parameters
        ----------
        batch : tuple
            The batch of data.
        batch_idx : int
            The index of the batch.

        Raises
        ------
        NotImplementedError
            Always; only training and ensemble validation are implemented
            for probabilistic forecasters.
        """
        raise NotImplementedError(
            "Ensemble test evaluation is not implemented for "
            "probabilistic forecasters."
        )
