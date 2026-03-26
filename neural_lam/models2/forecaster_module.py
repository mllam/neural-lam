# Standard library
import os
import warnings
from typing import Any, Dict, List

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr

# First-party
from neural_lam.utils import get_integer_time

# Local
from .. import metrics, vis
from ..datastore import BaseDatastore
from ..weather_dataset import WeatherDataset
from .forecaster import Forecaster


class ForecasterModule(pl.LightningModule):
    """
    Lightning wrapper that handles training/evaluation around a Forecaster.
    """

    # pylint: disable=arguments-differ
    def __init__(
        self,
        forecaster: Forecaster,
        args,
        config=None,
        datastore: BaseDatastore = None,
    ):
        super().__init__()

        # Backward compatibility for old positional order:
        # ForecasterModule(args, forecaster, datastore)
        if not hasattr(forecaster, "step_predictor") and hasattr(
            args, "step_predictor"
        ):
            old_args = forecaster
            old_forecaster = args
            old_datastore = config
            forecaster = old_forecaster
            args = old_args
            config = None
            datastore = old_datastore

        self.save_hyperparameters(ignore=["datastore", "forecaster", "config"])

        self.args = args
        self.forecaster = forecaster
        self.config = config
        self._datastore = datastore
        if self._datastore is None:
            raise ValueError("datastore must be provided to ForecasterModule")

        if not hasattr(self.forecaster, "step_predictor"):
            raise ValueError(
                "ForecasterModule currently expects forecaster.step_predictor"
            )

        # Instantiate loss function
        self.loss = metrics.get_metric(args.loss)

        self.val_metrics: Dict[str, List] = {
            "mse": [],
        }
        self.test_metrics: Dict[str, List] = {
            "mse": [],
            "mae": [],
        }
        if self.output_std:
            self.test_metrics["output_std"] = []  # Treat as metric

        # For making restoring of optimizer state optional
        self.restore_opt = args.restore_opt

        # For example plotting
        self.n_example_pred = args.n_example_pred
        self.plotted_examples = 0

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps: List[Any] = []

        self.time_step_int, self.time_step_unit = get_integer_time(
            self._datastore.step_length
        )

    @property
    def step_predictor(self):
        """
        Expose underlying one-step predictor.
        """
        return self.forecaster.step_predictor

    @property
    def output_std(self) -> bool:
        """
        Whether the step predictor returns predictive std.
        """
        return bool(self.step_predictor.output_std)

    @property
    def state_mean(self):
        """
        Shortcut to state mean buffer.
        """
        return self.step_predictor.state_mean

    @property
    def state_std(self):
        """
        Shortcut to state std buffer.
        """
        return self.step_predictor.state_std

    @property
    def interior_mask_bool(self):
        """
        Get the interior mask as a boolean (N,) mask.
        """
        return self.step_predictor.interior_mask[:, 0].to(torch.bool)

    def _create_dataarray_from_tensor(
        self,
        tensor: torch.Tensor,
        time: torch.Tensor,
        split: str,
        category: str,
    ) -> xr.DataArray:
        """
        Create an xr.DataArray matching the datastore coordinates.
        """
        # TODO: creating an instance of WeatherDataset here on every call is
        # not how this should be done but whether WeatherDataset should be
        # provided to ForecasterModule or where to put plotting still needs
        # discussion.
        weather_dataset = WeatherDataset(datastore=self._datastore, split=split)
        time = np.array(time.cpu(), dtype="datetime64[ns]")
        da = weather_dataset.create_dataarray_from_tensor(
            tensor=tensor, time=time, category=category
        )
        return da

    def configure_optimizers(self):
        """
        Configure optimizer.
        """
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.args.lr, betas=(0.9, 0.95)
        )
        return opt

    def common_step(self, batch):
        """
        Predict on a single batch.

        batch consists of:
        init_states: (B, 2, num_grid_nodes, d_features)
        target_states: (B, pred_steps, num_grid_nodes, d_features)
        forcing_features: (B, pred_steps, num_grid_nodes, d_forcing)
        """
        (init_states, target_states, forcing_features, batch_times) = batch

        prediction, pred_std = self.forecaster(
            init_states=init_states,
            forcing_features=forcing_features,
            true_states=target_states,
            ensemble_size=getattr(self.args, "ensemble_size", 1),
        )  # (B, pred_steps, num_grid_nodes, d_f)

        if pred_std is None and not self.output_std:
            pred_std = self.step_predictor.per_var_std  # (d_f,)

        return prediction, target_states, pred_std, batch_times

    def training_step(self, batch):
        """
        Train on single batch.
        """
        prediction, target, pred_std, _ = self.common_step(batch)

        # Compute loss
        batch_loss = torch.mean(
            self.loss(
                prediction, target, pred_std, mask=self.interior_mask_bool
            )
        )  # mean over unrolled times and batch

        log_dict = {"train_loss": batch_loss}
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return batch_loss

    def all_gather_cat(self, tensor_to_gather):
        """
        Gather tensors across all ranks and concatenate dim 0.
        """
        gathered = self.all_gather(tensor_to_gather)
        # all_gather adds a leading dim (K,) only on multi-device runs;
        # on single-device it returns the tensor unchanged.
        if gathered.dim() > tensor_to_gather.dim():
            return gathered.flatten(0, 1)
        return gathered

    # newer lightning versions requires batch_idx argument, even if unused
    # pylint: disable-next=unused-argument
    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch.
        """
        prediction, target, pred_std, _ = self.common_step(batch)

        time_step_loss = torch.mean(
            self.loss(
                prediction, target, pred_std, mask=self.interior_mask_bool
            ),
            dim=0,
        )  # (time_steps-1,)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.args.val_steps_to_log
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

        # Store MSEs
        entry_mses = metrics.mse(
            prediction,
            target,
            pred_std,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.val_metrics["mse"].append(entry_mses)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch.
        """
        self.aggregate_and_plot_metrics(self.val_metrics, prefix="val")
        for metric_list in self.val_metrics.values():
            metric_list.clear()

    # pylint: disable-next=unused-argument
    def test_step(self, batch, batch_idx):
        """
        Run test on single batch.
        """
        prediction, target, pred_std, batch_times = self.common_step(batch)
        _ = batch_times

        time_step_loss = torch.mean(
            self.loss(
                prediction, target, pred_std, mask=self.interior_mask_bool
            ),
            dim=0,
        )  # (time_steps-1,)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        test_log_dict = {
            f"test_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.args.val_steps_to_log
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

        # Compute all evaluation metrics for error maps.
        for metric_name in ("mse", "mae"):
            metric_func = metrics.get_metric(metric_name)
            batch_metric_vals = metric_func(
                prediction,
                target,
                pred_std,
                mask=self.interior_mask_bool,
                sum_vars=False,
            )  # (B, pred_steps, d_f)
            self.test_metrics[metric_name].append(batch_metric_vals)

        if self.output_std:
            # Store output std. per variable, spatially averaged
            mean_pred_std = torch.mean(
                pred_std[..., self.interior_mask_bool, :], dim=-2
            )  # (B, pred_steps, d_f)
            self.test_metrics["output_std"].append(mean_pred_std)

        # Save per-sample spatial loss for specific times
        spatial_loss = self.loss(
            prediction, target, pred_std, average_grid=False
        )  # (B, pred_steps, num_grid_nodes)
        log_spatial_losses = spatial_loss[
            :, [step - 1 for step in self.args.val_steps_to_log]
        ]
        self.spatial_loss_maps.append(log_spatial_losses)
        # (B, N_log, num_grid_nodes)

        # Plot example predictions (on rank 0 only)
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

    def plot_examples(self, batch, n_examples, split, prediction=None):
        """
        Plot first n_examples forecasts from a batch.
        """
        if prediction is None:
            prediction, target, _, _ = self.common_step(batch)
            _ = target

        target = batch[1]
        time = batch[3]

        # Rescale to original data scale
        prediction_rescaled = prediction * self.state_std + self.state_mean
        target_rescaled = target * self.state_std + self.state_mean

        # Iterate over the examples
        for pred_slice, target_slice, time_slice in zip(
            prediction_rescaled[:n_examples],
            target_rescaled[:n_examples],
            time[:n_examples],
        ):
            self.plotted_examples += 1

            da_prediction = self._create_dataarray_from_tensor(
                tensor=pred_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")
            da_target = self._create_dataarray_from_tensor(
                tensor=target_slice,
                time=time_slice,
                split=split,
                category="state",
            ).unstack("grid_index")

            var_vmin = (
                torch.minimum(
                    pred_slice.flatten(0, 1).min(dim=0)[0],
                    target_slice.flatten(0, 1).min(dim=0)[0],
                )
                .cpu()
                .numpy()
            )
            var_vmax = (
                torch.maximum(
                    pred_slice.flatten(0, 1).max(dim=0)[0],
                    target_slice.flatten(0, 1).max(dim=0)[0],
                )
                .cpu()
                .numpy()
            )
            var_vranges = list(zip(var_vmin, var_vmax))

            # Iterate over prediction horizon time steps
            for t_i, _ in enumerate(zip(pred_slice, target_slice), start=1):
                var_figs = [
                    vis.plot_prediction(
                        datastore=self._datastore,
                        title=f"{var_name}, t={t_i}"
                        f" ({self.time_step_int * t_i}"
                        f"{self.time_step_unit})",
                        colorbar_label=var_unit,
                        vrange=var_vrange,
                        da_prediction=da_prediction.isel(
                            state_feature=var_i, time=t_i - 1
                        ).squeeze(),
                        da_target=da_target.isel(
                            state_feature=var_i, time=t_i - 1
                        ).squeeze(),
                    )
                    for var_i, (var_name, var_unit, var_vrange) in enumerate(
                        zip(
                            self._datastore.get_vars_names("state"),
                            self._datastore.get_vars_units("state"),
                            var_vranges,
                        )
                    )
                ]

                example_i = self.plotted_examples

                for var_name, fig in zip(
                    self._datastore.get_vars_names("state"), var_figs
                ):
                    # Logger behavior differs between wandb and others.
                    if isinstance(self.logger, pl.loggers.WandbLogger):
                        key = f"{var_name}_example_{example_i}"
                    else:
                        key = f"{var_name}_example"

                    if hasattr(self.logger, "log_image"):
                        self.logger.log_image(key=key, images=[fig], step=t_i)
                    else:
                        warnings.warn(
                            f"{self.logger} does not support image logging."
                        )

                plt.close("all")

            # Save pred and target as .pt files
            torch.save(
                pred_slice.cpu(),
                os.path.join(
                    self.logger.save_dir,
                    f"example_pred_{self.plotted_examples}.pt",
                ),
            )
            torch.save(
                target_slice.cpu(),
                os.path.join(
                    self.logger.save_dir,
                    f"example_target_{self.plotted_examples}.pt",
                ),
            )

    def create_metric_log_dict(self, metric_tensor, prefix, metric_name):
        """
        Build metric figure dict for logger.
        """
        log_dict = {}
        metric_fig = vis.plot_error_map(
            errors=metric_tensor,
            datastore=self._datastore,
        )
        full_log_name = f"{prefix}_{metric_name}"
        log_dict[full_log_name] = metric_fig

        if prefix == "test":
            metric_fig.savefig(
                os.path.join(self.logger.save_dir, f"{full_log_name}.pdf")
            )
            np.savetxt(
                os.path.join(self.logger.save_dir, f"{full_log_name}.csv"),
                metric_tensor.cpu().numpy(),
                delimiter=",",
            )

        var_names = self._datastore.get_vars_names(category="state")
        if full_log_name in self.args.metrics_watch:
            for var_i, timesteps in self.args.var_leads_metrics_watch.items():
                var_name = var_names[var_i]
                for step in timesteps:
                    key = f"{full_log_name}_{var_name}_step_{step}"
                    log_dict[key] = metric_tensor[step - 1, var_i]

        return log_dict

    def aggregate_and_plot_metrics(self, metrics_dict, prefix):
        """
        Aggregate and create error map plots for all metrics in metrics_dict.
        """
        log_dict = {}
        for metric_name, metric_val_list in metrics_dict.items():
            metric_tensor = self.all_gather_cat(
                torch.cat(metric_val_list, dim=0)
            )  # (N_eval, pred_steps, d_f)

            if self.trainer.is_global_zero:
                metric_tensor_averaged = torch.mean(metric_tensor, dim=0)
                # Take square root after all averaging to change MSE to RMSE
                if "mse" in metric_name:
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name.replace("mse", "rmse")

                metric_rescaled = metric_tensor_averaged * self.state_std
                log_dict.update(
                    self.create_metric_log_dict(
                        metric_rescaled, prefix, metric_name
                    )
                )

        assert all(
            isinstance(key, str) and isinstance(value, plt.Figure)
            for key, value in log_dict.items()
        )

        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            current_epoch = self.trainer.current_epoch
            for key, figure in log_dict.items():
                if not isinstance(self.logger, pl.loggers.WandbLogger):
                    key = f"{key}-{current_epoch}"
                if hasattr(self.logger, "log_image"):
                    self.logger.log_image(key=key, images=[figure])

            plt.close("all")

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at end of test epoch.
        """
        self.aggregate_and_plot_metrics(self.test_metrics, prefix="test")

        spatial_loss_tensor = self.all_gather_cat(
            torch.cat(self.spatial_loss_maps, dim=0)
        )  # (N_test, N_log, num_grid_nodes)
        if self.trainer.is_global_zero:
            mean_spatial_loss = torch.mean(spatial_loss_tensor, dim=0)

            loss_map_figs = [
                vis.plot_spatial_error(
                    error=loss_map,
                    datastore=self._datastore,
                    title=f"Test loss, t={t_i} "
                    f"({(self.time_step_int * t_i)} {self.time_step_unit})",
                )
                for t_i, loss_map in zip(
                    self.args.val_steps_to_log, mean_spatial_loss
                )
            ]

            for i, fig in enumerate(loss_map_figs):
                key = "test_loss"
                if not isinstance(self.logger, pl.loggers.WandbLogger):
                    key = f"{key}_{i}"
                if hasattr(self.logger, "log_image"):
                    self.logger.log_image(key=key, images=[fig])

            pdf_loss_map_figs = [
                vis.plot_spatial_error(
                    error=loss_map, datastore=self._datastore
                )
                for loss_map in mean_spatial_loss
            ]
            pdf_loss_maps_dir = os.path.join(
                self.logger.save_dir, "spatial_loss_maps"
            )
            os.makedirs(pdf_loss_maps_dir, exist_ok=True)
            for t_i, fig in zip(self.args.val_steps_to_log, pdf_loss_map_figs):
                fig.savefig(os.path.join(pdf_loss_maps_dir, f"loss_t{t_i}.pdf"))
            torch.save(
                mean_spatial_loss.cpu(),
                os.path.join(self.logger.save_dir, "mean_spatial_loss.pt"),
            )

        self.spatial_loss_maps.clear()

    def on_load_checkpoint(self, checkpoint):
        """
        Perform any changes to state dict before loading checkpoint.
        """
        if not self.restore_opt:
            opt = self.configure_optimizers()
            checkpoint["optimizer_states"] = [opt.state_dict()]
