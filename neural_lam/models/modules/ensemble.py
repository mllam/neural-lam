"""Lightning module evaluating ensemble forecasters."""

# Standard library
import warnings
from typing import Any

# Third-party
import torch

# Local
from ... import metrics
from ...config import NeuralLAMConfig
from ...datastore.base import BaseRegularGridDatastore
from ..forecasters.ensemble import BaseEnsembleForecaster
from .base import BaseForecastingModule


class EnsembleForecastingModule(BaseForecastingModule):
    """
    Lightning module for forecasters whose forecast is an ensemble.

    Training is inherited unchanged from ``BaseForecastingModule``: the
    wrapped forecaster assembles its own training loss. Validation and
    testing are ensemble based instead of deterministic: an ensemble is
    sampled from the forecaster and its mean scored per lead time and
    variable, with validation additionally reporting the forecaster's own
    objective. The module only assumes that the forecaster returns ensemble
    forecasts of the correct shape; it makes no assumption on how the
    members are produced.
    """

    # The wrapped forecaster's forward must return an ensemble
    forecaster: BaseEnsembleForecaster

    def __init__(
        self,
        forecaster: BaseEnsembleForecaster,
        config: NeuralLAMConfig,
        datastore: BaseRegularGridDatastore,
        *,
        eval_ensemble_size: int,
        lr: float = 1e-3,
        restore_opt: bool = False,
        n_example_pred: int = 1,
        create_gif: bool = False,
        val_steps_to_log: list[int] | None = None,
        train_steps_to_log: list[int] | None = None,
        metrics_watch: list[str] | None = None,
        var_leads_metrics_watch: dict[int, list[int]] | None = None,
        args: Any | None = None,
    ) -> None:
        """
        Initialize the module and store the evaluation ensemble size.

        Parameters
        ----------
        forecaster : BaseEnsembleForecaster
            The forecaster to evaluate. Its ``forward`` must return an
            ensemble, since validation and testing score a sampled one.
        config : NeuralLAMConfig
            Configuration object for the neural LAM model.
        datastore : BaseRegularGridDatastore
            Datastore providing grid metadata and data access.
        eval_ensemble_size : int
            Number of ensemble members sampled during validation and
            testing. Keyword-only and required: how many members to draw is
            a choice this module cannot sensibly default.
        lr : float, default 1e-3
            Learning rate for the optimizer.
        restore_opt : bool, default False
            Whether to restore optimizer state from checkpoint.
        n_example_pred : int, default 1
            Number of example predictions to plot during testing. Unused
            here, since ``test_step`` plots no examples.
        create_gif : bool, default False
            Whether to create GIFs of example predictions. Unused here, for
            the same reason.
        val_steps_to_log : list of int, optional
            Specific predicted steps to log during validation/testing.
            Unused here: nothing this module reports decomposes per step,
            since the ensemble metrics are logged for every step at once as
            heatmaps and the objective is a single scalar.
        train_steps_to_log : list of int, optional
            Specific predicted steps to log during training. Has no effect
            unless the forecaster's objective decomposes per step; see
            ``BaseForecastingModule.training_step``.
        metrics_watch : list of str, optional
            List of metrics to watch and log specifically.
        var_leads_metrics_watch : dict of {int: list of int}, optional
            Mapping from variable index to a list of predicted steps to log
            individually for the configured metrics.
        args : argparse.Namespace, optional
            Pre-refactor ``ARModel`` checkpoint hyperparameters; see
            ``BaseForecastingModule.__init__``.

        Raises
        ------
        ValueError
            If ``eval_ensemble_size`` is less than 1.
        """
        super().__init__(
            forecaster=forecaster,
            config=config,
            datastore=datastore,
            lr=lr,
            restore_opt=restore_opt,
            n_example_pred=n_example_pred,
            create_gif=create_gif,
            val_steps_to_log=val_steps_to_log,
            train_steps_to_log=train_steps_to_log,
            metrics_watch=metrics_watch,
            var_leads_metrics_watch=var_leads_metrics_watch,
            args=args,
        )
        if eval_ensemble_size < 1:
            raise ValueError(
                "eval_ensemble_size must be at least 1, "
                f"got {eval_ensemble_size}"
            )
        # The base class names the hyperparameters it knows about; this one
        # is ours, so record it here. A second call merges rather than
        # replaces, so both end up saved.
        self.save_hyperparameters({"eval_ensemble_size": eval_ensemble_size})
        self.eval_ensemble_size = eval_ensemble_size
        self.val_metrics = {"ens_mse": []}
        self.test_metrics = {"ens_mse": []}

    def _ensemble_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Sample an ensemble and compute the per-variable MSE of its mean.

        Shared by ``validation_step`` and ``test_step``: samples
        ``self.eval_ensemble_size`` members and scores the ensemble mean
        with plain (unweighted) MSE on interior nodes. Only the squared
        errors are computed here; ``aggregate_and_plot_metrics`` reduces
        them to a per-lead per-variable RMSE once per epoch, since the
        square root has to be taken after averaging over every sample
        rather than per batch.

        This is a diagnostic metric, not the training loss: it always
        scores the ensemble mean with plain MSE, regardless of what
        objective ``compute_training_loss`` actually trains on. The
        objective itself is logged by ``_log_objective``.

        Parameters
        ----------
        batch : tuple
            The batch of data.

        Returns
        -------
        torch.Tensor
            Per-variable ensemble-mean MSE, shape
            ``(B, pred_steps, num_state_vars)``, for epoch-end aggregation.
        """
        init_states, target_states, forcing_features, _ = batch
        ensemble, _ = self.forecaster(
            init_states,
            forcing_features,
            target_states,
            num_members=self.eval_ensemble_size,
        )
        ensemble_mean = ensemble.mean(dim=1)

        entry_mses = metrics.mse(
            ensemble_mean,
            target_states,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, num_state_vars)

        return entry_mses

    def _log_objective(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """
        Log the forecaster's own training objective as ``val_mean_loss``.

        Named as ``DeterministicForecastingModule`` names it, so that
        ``ModelCheckpoint`` has a scalar to monitor. What that objective is
        stays entirely up to the forecaster, and it costs a forward pass of
        its own rather than being derived from the sampled ensemble, since
        the two need not agree on either the member count or the scoring
        rule. Validation only: nothing monitors the test phase, so paying
        that pass again there would buy nothing.

        Parameters
        ----------
        batch : tuple
            The batch of data.
        """
        init_states, target_states, forcing_features, _ = batch
        batch_loss, loss_components = self.forecaster.compute_training_loss(
            init_states,
            forcing_features,
            target_states,
            interior_mask_bool=self.interior_mask_bool,
        )

        log_dict = {
            f"val_{name}": value for name, value in loss_components.items()
        }
        log_dict["val_mean_loss"] = batch_loss
        self.log_dict(
            log_dict,
            on_epoch=True,
            sync_dist=True,
            batch_size=init_states.shape[0],
        )

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """
        Perform a single ensemble validation step.

        Logs the forecaster's objective as ``val_mean_loss``, scores the
        ensemble mean against the target states (see ``_ensemble_step``) and
        collects per-variable ensemble-mean MSE for epoch-end aggregation.

        Parameters
        ----------
        batch : tuple
            The batch of data.
        batch_idx : int
            The index of the batch.
        """
        # Note that we here do two forward passes: One for computing loss
        # and one for computing ensemble metrics. Required as computing loss
        # might not involve making a forecast the same way as during inference.
        self._log_objective(batch)
        entry_mses = self._ensemble_step(batch)
        self.val_metrics["ens_mse"].append(entry_mses)

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
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
        entry_mses = self._ensemble_step(batch)
        self.test_metrics["ens_mse"].append(entry_mses)

    def on_test_epoch_end(self) -> None:
        """
        Perform actions at the end of the test epoch.

        Aggregates the ensemble test metrics. Implements
        ``BaseForecastingModule.on_test_epoch_end`` without the spatial loss
        maps and example plots that ``DeterministicForecastingModule`` adds,
        since ``test_step`` here does not populate them.
        """
        self.aggregate_and_plot_metrics(self.test_metrics, prefix="test")

        metrics_watch = (
            self.hparams.metrics_watch  # ty: ignore[unresolved-attribute]
        )
        if self.trainer.is_global_zero and metrics_watch:
            unmatched = set(metrics_watch) - self.matched_metrics
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
