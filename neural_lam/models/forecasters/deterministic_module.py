"""Lightning module evaluating forecasters through a single deterministic
rollout per batch."""

# Standard library
import os
import warnings
from typing import Any

# Third-party
import pytorch_lightning as pl
import torch

# Local
from ... import metrics, vis
from .base_module import BaseForecasterModule


class DeterministicForecasterModule(BaseForecasterModule):
    """
    Lightning module for a single deterministic forecast per batch.

    Validation and testing score the forecaster's own single-rollout
    prediction via ``forecaster.score``, as opposed to
    ``ProbabilisticForecasterModule``, which samples and scores an
    ensemble. Training is shared with that module unchanged (see
    ``BaseForecasterModule.training_step``).
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the module and its deterministic evaluation metrics.

        Parameters
        ----------
        *args
            Positional arguments forwarded to
            ``BaseForecasterModule.__init__`` (``forecaster``, ``config``,
            ``datastore``, ...).
        **kwargs
            Keyword arguments forwarded to ``BaseForecasterModule.__init__``
            (``lr``, ...).
        """
        super().__init__(*args, **kwargs)
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
        prediction, target_states, pred_std, _ = self.common_step(batch)

        time_step_loss = torch.mean(
            self.forecaster.score(
                prediction,
                target_states,
                pred_std,
                mask=self.interior_mask_bool,
            ),
            dim=0,
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

        entry_mses = self.forecaster.score(
            prediction,
            target_states,
            pred_std,
            metric=metrics.mse,
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
        prediction, target_states, pred_std, _ = self.common_step(batch)

        if pred_std is not None:
            mean_pred_std = torch.mean(
                pred_std[..., self.interior_mask_bool, :], dim=-2
            )
            self.test_metrics["output_std"].append(mean_pred_std)

        time_step_loss = torch.mean(
            self.forecaster.score(
                prediction,
                target_states,
                pred_std,
                mask=self.interior_mask_bool,
            ),
            dim=0,
        )
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

        for metric_name in ("mse", "mae"):
            batch_metric_vals = self.forecaster.score(
                prediction,
                target_states,
                pred_std,
                metric=metrics.get_metric(metric_name),
                mask=self.interior_mask_bool,
                sum_vars=False,
            )
            self.test_metrics[metric_name].append(batch_metric_vals)

        spatial_loss = self.forecaster.score(
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
