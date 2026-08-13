"""Lightning module evaluating forecasters through a single deterministic
forecast per batch."""

# Standard library
import os
import warnings
from typing import Any

# Third-party
import pytorch_lightning as pl
import torch

# Local
from ... import metrics, vis
from ...config import NeuralLAMConfig
from ...datastore import BaseDatastore
from ..forecasters.deterministic import DeterministicForecaster
from .base import BaseForecastingModule


class DeterministicForecastingModule(BaseForecastingModule):
    """
    Lightning module for a single deterministic forecast per batch.

    Validation and testing evaluate the forecaster's own single prediction,
    as opposed to ``ProbabilisticForecastingModule``, which samples and scores
    an ensemble. Training is shared with that module unchanged (see
    ``BaseForecastingModule.training_step``).

    The reported loss comes from ``forecaster.compute_loss_from_forecast``,
    since only the forecaster knows its objective. The reported metrics
    (mse, mae) are
    computed here from ``neural_lam.metrics``: they are fixed regardless of
    what the forecaster trains on, so routing them through it would add a
    layer without adding meaning.
    """

    # Narrowed from Forecaster: this module calls
    # forecaster.compute_loss_from_forecast()
    forecaster: DeterministicForecaster

    def __init__(
        self,
        forecaster: DeterministicForecaster,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        lr: float = 1e-3,
        restore_opt: bool = False,
        n_example_pred: int = 1,
        create_gif: bool = False,
        val_steps_to_log: list[int] | None = None,
        train_steps_to_log: list[int] | None = None,
        metrics_watch: list[str] | None = None,
        var_leads_metrics_watch: dict[int, list[int]] | None = None,
        args=None,
    ):
        """
        Initialize the module and its deterministic evaluation metrics.

        Parameters
        ----------
        forecaster : DeterministicForecaster
            The forecaster to evaluate. Must supply
            ``compute_loss_from_forecast``, i.e. carry the deterministic
            objective, since validation and testing score a single
            prediction through it.
        config : NeuralLAMConfig
            Configuration object for the neural LAM model.
        datastore : BaseDatastore
            Datastore providing grid metadata and data access.
        lr : float, default 1e-3
            Learning rate for the optimizer.
        restore_opt : bool, default False
            Whether to restore optimizer state from checkpoint.
        n_example_pred : int, default 1
            Number of example predictions to plot during testing.
        create_gif : bool, default False
            Whether to create GIFs of example predictions.
        val_steps_to_log : list of int, optional
            Specific predicted steps to log during validation/testing.
        train_steps_to_log : list of int, optional
            Specific predicted steps to log during training, reported as
            ``train_loss_unroll{i}`` (see ``training_step``).
        metrics_watch : list of str, optional
            List of metrics to watch and log specifically.
        var_leads_metrics_watch : dict of {int: list of int}, optional
            Mapping from variable index to a list of predicted steps to log
            individually for the configured metrics.
        args : argparse.Namespace, optional
            Pre-refactor ``ARModel`` checkpoint hyperparameters; see
            ``BaseForecastingModule.__init__``.
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
        self.val_metrics: dict[str, list] = {
            "mse": [],
        }
        self.test_metrics: dict[str, list] = {
            "mse": [],
            "mae": [],
        }
        if self.forecaster.predicts_std:
            self.test_metrics["output_std"] = []  # Treat as metric

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps: list[Any] = []

    def training_step(self, batch):
        """
        Perform a single training step, logging the per-step breakdown.

        Overrides ``BaseForecastingModule.training_step``, which logs only the
        scalar objective because a forecaster's training loss need not
        decompose over predicted steps. The deterministic objective does, so
        ``--train_steps_to_log`` can report individual steps as
        ``train_loss_unroll{i}`` and the logged ``train_loss`` is their mean,
        equal to ``forecaster.compute_training_loss``.

        Parameters
        ----------
        batch : tuple
            The batch of data.

        Returns
        -------
        torch.Tensor
            The computed loss for the training step.
        """
        _, _, _, time_step_loss = self._compute_prediction_and_loss(batch)
        batch_loss = torch.mean(time_step_loss)
        self._log_step_loss(
            time_step_loss, batch_loss, "train", batch[0].shape[0]
        )
        return batch_loss

    def _compute_prediction_and_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute predicted mean, standard deviation, and step-wise loss.
        Also extract and return corresponding target from batch.

        Shared by ``training_step``, ``validation_step`` and ``test_step``.
        The scoring rule applied here is the forecaster's own, so the mean
        of the per-step losses is its training objective; a forecaster whose
        objective is not a per-step scoring rule needs the plain
        ``BaseForecastingModule.training_step`` instead.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            The batch of data.

        Returns
        -------
        prediction : torch.Tensor
            Model predictions, shape
            ``(B, pred_steps, num_grid_nodes, num_state_vars)``.
        target_states : torch.Tensor
            Target states, shape
            ``(B, pred_steps, num_grid_nodes, num_state_vars)``.
        pred_std : torch.Tensor
            Predicted or pre-defined standard deviation, shape
            ``(B, pred_steps, num_grid_nodes, num_state_vars)`` or
            ``(num_state_vars,)``.
        time_step_loss : torch.Tensor
            Loss for each unroll step, shape ``(pred_steps,)``.
        """
        prediction, target_states, pred_std, _ = self.common_step(batch)

        time_step_loss = torch.mean(
            self.forecaster.compute_loss_from_forecast(
                prediction,
                target_states,
                pred_std,
                mask=self.interior_mask_bool,
            ),
            dim=0,
        )
        return prediction, target_states, pred_std, time_step_loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Parameters
        ----------
        batch : tuple
            The batch of data.
        batch_idx : int
            The index of the batch.
        """
        prediction, target_states, pred_std, time_step_loss = (
            self._compute_prediction_and_loss(batch)
        )

        mean_loss = torch.mean(time_step_loss)
        self._warn_skipped_val_steps(len(time_step_loss), "val")

        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.hparams.val_steps_to_log
            if step <= len(time_step_loss)
        }
        val_log_dict["val_mean_loss"] = mean_loss
        self.log_dict(
            val_log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )

        # Reported independently of the training objective, so computed here
        # rather than through the forecaster
        entry_mses = metrics.mse(
            prediction,
            target_states,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )
        self.val_metrics["mse"].append(entry_mses)

    # pylint: disable-next=unused-argument
    def test_step(self, batch, batch_idx):
        """
        Perform a single test step.

        Parameters
        ----------
        batch : tuple
            The batch of data.
        batch_idx : int
            The index of the batch.
        """
        prediction, target_states, pred_std, time_step_loss = (
            self._compute_prediction_and_loss(batch)
        )

        if pred_std is not None:
            mean_pred_std = torch.mean(
                pred_std[..., self.interior_mask_bool, :], dim=-2
            )
            self.test_metrics["output_std"].append(mean_pred_std)

        mean_loss = torch.mean(time_step_loss)
        self._warn_skipped_val_steps(len(time_step_loss), "test")

        test_log_dict = {
            f"test_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.hparams.val_steps_to_log
            if step <= len(time_step_loss)
        }
        test_log_dict["test_mean_loss"] = mean_loss

        self.log_dict(
            test_log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )

        # Reported independently of the training objective, so computed here
        # rather than through the forecaster
        for metric_name in ("mse", "mae"):
            batch_metric_vals = metrics.get_metric(metric_name)(
                prediction,
                target_states,
                mask=self.interior_mask_bool,
                sum_vars=False,
            )
            self.test_metrics[metric_name].append(batch_metric_vals)

        spatial_loss = self.forecaster.compute_loss_from_forecast(
            prediction, target_states, pred_std, average_grid=False
        )
        log_spatial_losses = spatial_loss[
            :,
            [
                step - 1
                for step in self.hparams.val_steps_to_log
                if step <= spatial_loss.shape[1]
            ],
        ]
        self.spatial_loss_maps.append(log_spatial_losses)

        if (
            self.trainer.is_global_zero
            and self.plotted_examples < self.n_example_pred
        ):
            n_additional_examples = min(
                prediction.shape[0],
                self.n_example_pred - self.plotted_examples,
            )

            self.plot_examples(
                batch,
                n_additional_examples,
                prediction=prediction,
                split="test",
            )

    def on_test_epoch_end(self):
        """
        Perform actions at the end of the test epoch.
        Aggregates and plots test metrics and spatial loss maps.
        """
        self.aggregate_and_plot_metrics(self.test_metrics, prefix="test")

        spatial_loss_tensor = self.all_gather_cat(
            torch.cat(self.spatial_loss_maps, dim=0)
        )
        if self.trainer.is_global_zero:
            mean_spatial_loss = torch.mean(spatial_loss_tensor, dim=0)

            loss_map_figs = [
                vis.plot_spatial_error(
                    error=loss_map,
                    datastore=self.datastore,
                    title=f"Test loss, t={t_i} "
                    f"({(self.time_step_int * t_i)} {self.time_step_unit})",
                )
                for t_i, loss_map in zip(
                    self.hparams.val_steps_to_log, mean_spatial_loss
                )
            ]

            for i, fig in enumerate(loss_map_figs):
                key = "test_loss"
                if not isinstance(self.logger, pl.loggers.WandbLogger):
                    key = f"{key}_{i}"
                if hasattr(self.logger, "log_image"):
                    self.logger.log_image(key=key, images=[fig])

            pdf_loss_map_figs = [
                vis.plot_spatial_error(error=loss_map, datastore=self.datastore)
                for loss_map in mean_spatial_loss
            ]
            pdf_loss_maps_dir = os.path.join(
                self.logger.save_dir, "spatial_loss_maps"
            )
            os.makedirs(pdf_loss_maps_dir, exist_ok=True)
            for t_i, fig in zip(
                self.hparams.val_steps_to_log, pdf_loss_map_figs
            ):
                fig.savefig(os.path.join(pdf_loss_maps_dir, f"loss_t{t_i}.pdf"))

            torch.save(
                mean_spatial_loss.cpu(),
                os.path.join(self.logger.save_dir, "mean_spatial_loss.pt"),
            )

            if self.hparams.metrics_watch:
                unmatched = (
                    set(self.hparams.metrics_watch) - self.matched_metrics
                )
                if unmatched:
                    warnings.warn(
                        "The following metrics in --metrics_watch "
                        "were not found during test phase: "
                        f"{sorted(unmatched)}. Ensure the metric prefix "
                        "matches the evaluation mode (expected 'test_')."
                    )

        self.matched_metrics = set()
        self.spatial_loss_maps.clear()

        # Clear stored test metrics so repeated `trainer.test()` calls on
        # the same model instance start from a clean slate (otherwise the
        # tensors accumulate and skew the aggregated metrics).
        for metric_list in self.test_metrics.values():
            metric_list.clear()

        # Reset the example-plot counter so example prediction plots are
        # generated again on every `trainer.test()` call, not just the
        # first one (the guard `plotted_examples < n_example_pred` would
        # otherwise stay permanently False).
        self.plotted_examples = 0
