# Standard library
import os
import pickle
from typing import List, Union

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
import xarray as xr
from loguru import logger

# Local
from .. import metrics, vis
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..datastore.base import BaseRegularGridDatastore
from ..loss_weighting import get_state_feature_weighting
from ..weather_dataset import WeatherDataset


class ARModel(pl.LightningModule):
    """
    Generic auto-regressive weather model.
    Abstract class that can be extended.
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
        datastore_boundary: Union[BaseDatastore, None],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["datastore"])
        self.args = args
        self._datastore = datastore
        num_state_vars = datastore.get_num_data_vars(category="state")
        num_forcing_vars = datastore.get_num_data_vars(category="forcing")

        num_past_forcing_steps = args.num_past_forcing_steps
        num_future_forcing_steps = args.num_future_forcing_steps

        # Load static features for interior
        da_static_features = datastore.get_dataarray(
            category="static", split=None, standardize=True
        )
        self.register_buffer(
            "interior_static_features",
            torch.tensor(da_static_features.values, dtype=torch.float32),
            persistent=False,
        )

        # Load stats for rescaling and weights
        da_state_stats = datastore.get_standardization_dataarray(
            category="state"
        )
        state_stats = {
            "state_mean": torch.tensor(
                da_state_stats.state_mean.values, dtype=torch.float32
            ),
            "state_std": torch.tensor(
                da_state_stats.state_std.values, dtype=torch.float32
            ),
            # Change stats below to be for diff of standardized variables
            "diff_mean": torch.tensor(
                da_state_stats.state_diff_mean.values
                / da_state_stats.state_std.values,
                dtype=torch.float32,
            ),
            "diff_std": torch.tensor(
                da_state_stats.state_diff_std.values
                / da_state_stats.state_std.values,
                dtype=torch.float32,
            ),
        }

        for key, val in state_stats.items():
            self.register_buffer(key, val, persistent=False)

        state_feature_weights = get_state_feature_weighting(
            config=config, datastore=datastore
        )
        self.feature_weights = torch.tensor(
            state_feature_weights, dtype=torch.float32
        )

        # Double grid output dim. to also output std.-dev.
        self.output_std = bool(args.output_std)
        if self.output_std:
            # Pred. dim. in grid cell
            self.grid_output_dim = 2 * num_state_vars
        else:
            # Pred. dim. in grid cell
            self.grid_output_dim = num_state_vars
            # Store constant per-variable std.-dev. weighting
            # NOTE that this is the inverse of the multiplicative weighting
            # in wMSE/wMAE
            self.register_buffer(
                "per_var_std",
                self.diff_std / torch.sqrt(self.feature_weights),
                persistent=False,
            )

        # interior from data + static
        (
            self.num_interior_nodes,
            interior_static_dim,
        ) = self.interior_static_features.shape
        self.num_total_grid_nodes = self.num_interior_nodes
        self.interior_dim = (
            2 * self.grid_output_dim
            + interior_static_dim
            + num_forcing_vars
            * (num_past_forcing_steps + num_future_forcing_steps + 1)
        )

        # If datastore_boundary is given, the model is forced from the boundary
        self.boundary_forced = datastore_boundary is not None

        if self.boundary_forced:
            # Load static features for boundary
            da_boundary_static_features = datastore_boundary.get_dataarray(
                category="static", split=None, standardize=True
            )
            self.register_buffer(
                "boundary_static_features",
                torch.tensor(
                    da_boundary_static_features.values, dtype=torch.float32
                ),
                persistent=False,
            )

            # Compute dimensionalities (e.g. to instantiate MLPs)
            (
                self.num_boundary_nodes,
                boundary_static_dim,
            ) = self.boundary_static_features.shape

            # Compute boundary input dim separately
            num_boundary_forcing_vars = datastore_boundary.get_num_data_vars(
                category="forcing"
            )

            # Dimensionality of encoded time deltas
            self.time_delta_enc_dim = (
                args.hidden_dim
                if args.time_delta_enc_dim is None
                else args.time_delta_enc_dim
            )
            assert self.time_delta_enc_dim % 2 == 0, (
                "Number of dimensions to use for time delta encoding must be "
                "even (sin and cos)"
            )

            num_past_boundary_steps = args.num_past_boundary_steps
            num_future_boundary_steps = args.num_future_boundary_steps
            self.boundary_dim = (
                boundary_static_dim
                # Time delta counts as one additional forcing_feature
                + (num_boundary_forcing_vars + self.time_delta_enc_dim)
                * (num_past_boundary_steps + num_future_boundary_steps + 1)
            )
            # How many of the last boundary forcing dims contain time-deltas
            self.boundary_time_delta_dims = (
                num_past_boundary_steps + num_future_boundary_steps + 1
            )

            self.num_total_grid_nodes += self.num_boundary_nodes

        # Instantiate loss function
        self.loss = metrics.get_metric(args.loss)

        self.val_metrics = {
            "mse": [],
        }
        self.test_metrics = {
            "mse": [],
            "mae": [],
        }
        if self.output_std:
            self.test_metrics["output_std"] = []  # Treat as metric

        # For example plotting
        self.n_example_pred = args.n_example_pred
        self.plotted_examples = 0

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps = []

        # Set if grad checkpointing function should be used during rollout
        if args.grad_checkpointing:
            # Perform gradient checkpointing at each unrolling step
            self.unroll_ckpt_func = (
                lambda f, *args: torch.utils.checkpoint.checkpoint(
                    f, *args, use_reentrant=False
                )
            )
        else:
            self.unroll_ckpt_func = lambda f, *args: f(*args)

        # Store step length (h), taking subsampling into account
        self.step_length = datastore.step_length * args.interior_subsample_step

        # Make WeatherDataset:s for being able to make tensor into xr.DA
        # Note: Unclear if it is actually necessary to make one per split?
        # TODO: creating an instance of WeatherDataset here on is
        # not how this should be done but whether WeatherDataset should be
        # provided to ARModel or where to put plotting still needs discussion
        self.wds_for_da = {
            split: WeatherDataset(
                datastore=self._datastore,
                datastore_boundary=None,
                split=split,
            )
            for split in ("train", "val", "test")
        }

        # Which variables to plot during eval
        self.plot_vars = args.plot_vars
        for var in self.plot_vars:
            assert var in self._datastore.get_vars_names(
                "state"
            ), f"Can not plot variable {var}: not in datastore"

    def _create_dataarray_from_tensor(
        self,
        tensor: torch.Tensor,
        time: Union[int, List[int]],
        split: str,
        category: str,
    ) -> xr.DataArray:
        """
        Create an `xr.DataArray` from a tensor, with the correct dimensions and
        coordinates to match the datastore used by the model. This function in
        in effect is the inverse of what is returned by
        `WeatherDataset.__getitem__`.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to convert to a `xr.DataArray` with dimensions [time,
            grid_index, feature]. The tensor will be copied to the CPU if it is
            not already there.
        time : Union[int,List[int]]
            The time index or indices for the data, given as integers or a list
            of integers representing epoch time in nanoseconds. The ints will be
            copied to the CPU memory if they are not already there.
        split : str
            The split of the data, either 'train', 'val', or 'test'
        category : str
            The category of the data, either 'state' or 'forcing'
        """
        # Move to CPU if on GPU
        time = time.detach().cpu()
        time = np.array(time, dtype="datetime64[ns]")

        tensor = tensor.detach().cpu()
        weather_dataset = self.wds_for_da[split]
        da = weather_dataset.create_dataarray_from_tensor(
            tensor=tensor, time=time, category=category
        )
        return da

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.args.lr, betas=(0.9, 0.95)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.args.epochs,
            eta_min=self.args.min_lr if hasattr(self.args, "min_lr") else 0.0,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @staticmethod
    def expand_to_batch(x, batch_size):
        """
        Expand tensor with initial batch dimension
        """
        return x.unsqueeze(0).expand(batch_size, -1, -1)

    def predict_step(
        self, prev_state, prev_prev_state, forcing, boundary_forcing
    ):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_interior_nodes, feature_dim), X_t
        prev_prev_state: (B, num_interior_nodes, feature_dim), X_{t-1}
        forcing: (B, num_interior_nodes, forcing_dim)
        boundary_forcing: (B, num_boundary_nodes, boundary_forcing_dim)
        """
        raise NotImplementedError("No prediction step implemented")

    def unroll_prediction(self, init_states, forcing, boundary_forcing):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, 2, num_interior_nodes, d_f)
        forcing: (B, pred_steps, num_interior_nodes, d_static_f)
        boundary_forcing: (B, pred_steps, num_boundary_nodes, d_boundary_f)
        """
        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []
        pred_steps = forcing.shape[1]

        for i in range(pred_steps):
            forcing_step = forcing[:, i]

            if self.boundary_forced:
                boundary_forcing_step = boundary_forcing[:, i]
            else:
                boundary_forcing_step = None

            pred_state, pred_std = self.unroll_ckpt_func(
                self.predict_step,
                prev_state,
                prev_prev_state,
                forcing_step,
                boundary_forcing_step,
            )
            # state: (B, num_interior_nodes, d_f)
            # pred_std: (B, num_interior_nodes, d_f) or None

            prediction_list.append(pred_state)

            if self.output_std:
                pred_std_list.append(pred_std)

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = pred_state

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_interior_nodes, d_f)
        if self.output_std:
            pred_std = torch.stack(
                pred_std_list, dim=1
            )  # (B, pred_steps, num_interior_nodes, d_f)
        else:
            pred_std = self.per_var_std  # (d_f,)

        return prediction, pred_std

    def common_step(self, batch):
        """
        Predict on single batch
        batch consists of:
        init_states: (B, 2, num_interior_nodes, d_features)
        target_states: (B, pred_steps, num_interior_nodes, d_features)
        forcing: (B, pred_steps, num_interior_nodes, d_forcing),
        boundary_forcing:
            (B, pred_steps, num_boundary_nodes, d_boundary_forcing),
            where index 0 corresponds to index 1 of init_states
        """
        (
            init_states,
            target_states,
            forcing,
            boundary_forcing,
            batch_times,
        ) = batch

        prediction, pred_std = self.unroll_prediction(
            init_states, forcing, boundary_forcing
        )  # (B, pred_steps, num_interior_nodes, d_f)
        # prediction: (B, pred_steps, num_interior_nodes, d_f) pred_std: (B,
        # pred_steps, num_interior_nodes, d_f) or (d_f,)

        return prediction, target_states, pred_std, batch_times

    def training_step(self, batch):
        """
        Train on single batch
        """
        prediction, target, pred_std, _ = self.common_step(batch)

        # Compute loss - mean over unrolled times and batch
        batch_loss = torch.mean(
            self.loss(
                prediction,
                target,
                pred_std,
            )
        )

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
        Gather tensors across all ranks, and concatenate across dim. 0 (instead
        of stacking in new dim. 0)

        tensor_to_gather: (d1, d2, ...), distributed over K ranks

        returns: (K*d1, d2, ...)
        """
        return self.all_gather(tensor_to_gather).flatten(0, 1)

    # newer lightning versions requires batch_idx argument, even if unused
    # pylint: disable-next=unused-argument
    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch
        """
        prediction, target, pred_std, _ = self.common_step(batch)

        time_step_loss = torch.mean(
            self.loss(
                prediction,
                target,
                pred_std,
            ),
            dim=0,
        )  # (time_steps-1)
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
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.val_metrics["mse"].append(entry_mses)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """
        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.val_metrics, prefix="val")

        # Clear lists with validation metrics values
        for metric_list in self.val_metrics.values():
            metric_list.clear()

    def _save_predictions_to_zarr(
        self,
        batch_times: torch.Tensor,
        batch_predictions: torch.Tensor,
        batch_idx: int,
        zarr_output_path: str,
    ):
        """
        Save state predictions for single batch to zarr dataset. Will append to
        existing dataset for batch_idx > 0. Resulting dataset will contain a
        variable named `state` with coordinates (start_time,
        elapsed_forecast_duration, grid_index, state_feature).
        Parameters
        ----------
        batch_times : torch.Tensor[int]
            The times for the batch, given as epoch time in nanoseconds. Shape
            is (B, args.pred_steps) where B is the batch size and
            args.pred_steps is the number of prediction steps.
        batch_predictions : torch.Tensor[float]
            The predictions for the batch, given as (B, args.pred_steps,
            num_grid_nodes, d_f) where B is the batch size, args.pred_steps is
            the number of prediction steps, num_grid_nodes is the number of
            grid nodes, and d_f is the number of state features.
        batch_idx : int
            The index of the batch in the current epoch.
        """
        # Scale predictions back to original data scale
        batch_predictions_rescaled = (
            batch_predictions * self.state_std + self.state_mean
        )

        # Convert predictions to DataArray using _create_dataarray_from_tensor
        das_pred = []
        for i in range(len(batch_times)):
            da_pred = self._create_dataarray_from_tensor(
                tensor=batch_predictions_rescaled[i],
                time=batch_times[i],
                split="test",
                category="state",
            )
            # Unstack grid coords if necessary, this also avoids the need to
            # try to store a MultiIndex zarr dataset which is not supported by
            # xarray
            if isinstance(self._datastore, BaseRegularGridDatastore):
                da_pred = self._datastore.unstack_grid_coords(da_pred)

            # First entry in da_pred.coords["time"] is time of first prediction,
            # so init time of forecast is one time step before
            t0 = da_pred.coords["time"].values[0] - np.array(
                self.step_length, dtype="timedelta64[h]"
            )
            da_pred.coords["start_time"] = t0
            da_pred.coords["elapsed_forecast_duration"] = da_pred.time - t0
            da_pred = da_pred.swap_dims({"time": "elapsed_forecast_duration"})
            da_pred.name = "state"
            das_pred.append(da_pred)

        da_pred_batch = xr.concat(das_pred, dim="start_time")

        # Apply chunking start_time and elapsed_forecast_duration, but leave
        # whole state in one chunk
        da_pred_batch = da_pred_batch.chunk(
            {"start_time": 1, "elapsed_forecast_duration": 1}
        )

        if batch_idx == 0:
            logger.info(f"Saving predictions to {zarr_output_path}")
            da_pred_batch.to_zarr(
                zarr_output_path,
                mode="w",
                consolidated=True,
                encoding={
                    "start_time": {
                        "units": "Seconds since 1970-01-01 00:00:00",
                        "dtype": "int64",
                    },
                },
            )
        else:
            da_pred_batch.to_zarr(
                zarr_output_path, mode="a", append_dim="start_time"
            )

    # pylint: disable-next=unused-argument
    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        """
        # TODO Here batch_times can be used for plotting routines
        prediction, target, pred_std, batch_times = self.common_step(batch)
        # prediction: (B, pred_steps, num_interior_nodes, d_f) pred_std: (B,
        # pred_steps, num_interior_nodes, d_f) or (d_f,)

        time_step_loss = torch.mean(
            self.loss(
                prediction,
                target,
                pred_std,
            ),
            dim=0,
        )  # (time_steps-1,)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        test_log_dict = {
            f"test_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.args.val_steps_to_log
        }
        test_log_dict["test_mean_loss"] = mean_loss

        self.log_dict(
            test_log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )

        # Compute all evaluation metrics for error maps Note: explicitly list
        # metrics here, as test_metrics can contain additional ones, computed
        # differently, but that should be aggregated on_test_epoch_end
        for metric_name in ("mse", "mae"):
            metric_func = metrics.get_metric(metric_name)
            batch_metric_vals = metric_func(
                prediction,
                target,
                pred_std,
                sum_vars=False,
            )  # (B, pred_steps, d_f)
            self.test_metrics[metric_name].append(batch_metric_vals)

        if self.output_std:
            # Store output std. per variable, spatially averaged
            mean_pred_std = torch.mean(pred_std, dim=-2)  # (B, pred_steps, d_f)
            self.test_metrics["output_std"].append(mean_pred_std)

        # Save per-sample spatial loss for specific times
        spatial_loss = self.loss(
            prediction, target, pred_std, average_grid=False
        )  # (B, pred_steps, num_interior_nodes)
        log_spatial_losses = spatial_loss[
            :, [step - 1 for step in self.args.val_steps_to_log]
        ]
        self.spatial_loss_maps.append(log_spatial_losses)
        # (B, N_log, num_interior_nodes)

        if self.args.save_eval_to_zarr_path:
            self._save_predictions_to_zarr(
                batch_times=batch_times,
                batch_predictions=prediction,
                batch_idx=batch_idx,
                zarr_output_path=self.args.save_eval_to_zarr_path,
            )

        # Plot example predictions (on rank 0 only)
        if (
            self.trainer.is_global_zero
            and self.plotted_examples < self.n_example_pred
        ):
            # Need to plot more example predictions
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
        Plot the first n_examples forecasts from batch

        batch: batch with data to plot corresponding forecasts for
        n_examples: number of forecasts to plot
        prediction: (B, pred_steps, num_interior_nodes, d_f),
            existing prediction. Generate if None.
        """
        if prediction is None:
            prediction, target, _, _ = self.common_step(batch)

        target = batch[1]
        time = batch[-1]

        # Rescale to original data scale
        prediction_rescaled = prediction * self.state_std + self.state_mean
        target_rescaled = target * self.state_std + self.state_mean

        # Iterate over the examples
        for pred_slice, target_slice, time_slice in zip(
            prediction_rescaled[:n_examples],
            target_rescaled[:n_examples],
            time[:n_examples],
        ):
            # Each slice is (pred_steps, num_interior_nodes, d_f)
            self.plotted_examples += 1  # Increment already here

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
            )  # (d_f,)
            var_vmax = (
                torch.maximum(
                    pred_slice.flatten(0, 1).max(dim=0)[0],
                    target_slice.flatten(0, 1).max(dim=0)[0],
                )
                .cpu()
                .numpy()
            )  # (d_f,)
            var_vranges = list(zip(var_vmin, var_vmax))

            # Iterate over prediction horizon time steps
            for t_i, _ in enumerate(zip(pred_slice, target_slice), start=1):
                # Create one figure per plot variable at this time step
                var_figs = {
                    var_name: vis.plot_prediction(
                        datastore=self._datastore,
                        title=f"{var_name} ({var_unit}), "
                        f"t={t_i} ({self.step_length * t_i} h)",
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
                    if var_name in self.plot_vars
                }

                example_i = self.plotted_examples

                wandb.log(
                    {
                        f"{var_name}_example_{example_i}": wandb.Image(fig)
                        for var_name, fig in var_figs.items()
                    }
                )
                plt.close(
                    "all"
                )  # Close all figs for this time step, saves memory

    def create_metric_log_dict(self, metric_tensor, prefix, metric_name):
        """
        Put together a dict with everything to log for one metric. Also saves
        plots as pdf and csv if using test prefix.

        metric_tensor: (pred_steps, d_f), metric values per time and variable
        prefix: string, prefix to use for logging metric_name: string, name of
        the metric

        Return: log_dict: dict with everything to log for given metric
        """
        log_dict = {}
        metric_fig = vis.plot_error_map(
            errors=metric_tensor,
            datastore=self._datastore,
        )
        full_log_name = f"{prefix}_{metric_name}"
        log_dict[full_log_name] = wandb.Image(metric_fig)

        if prefix == "test":
            # Save pdf
            metric_fig.savefig(
                os.path.join(wandb.run.dir, f"{full_log_name}.pdf")
            )

        # Check if metrics are watched, log exact values for specific vars
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
        Aggregate and create error map plots for all metrics in metrics_dict

        metrics_dict: dictionary with metric_names and list of tensors
            with step-evals.
        prefix: string, prefix to use for logging
        """
        log_dict = {}
        xr_data_vars = {}
        for metric_name, metric_val_list in metrics_dict.items():
            metric_tensor = self.all_gather_cat(
                torch.cat(metric_val_list, dim=0)
            )  # (N_eval, pred_steps, d_f)

            if self.trainer.is_global_zero:
                metric_tensor_averaged = torch.mean(metric_tensor, dim=0)
                # (pred_steps, d_f)

                # Take square root after all averaging to change MSE to RMSE
                if "mse" in metric_name:
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name.replace("mse", "rmse")

                # NOTE: we here assume rescaling for all metrics is linear
                metric_rescaled = metric_tensor_averaged * self.state_std
                # (pred_steps, d_f)

                # Add to log dict
                log_dict.update(
                    self.create_metric_log_dict(
                        metric_rescaled, prefix, metric_name
                    )
                )

                # Add to xr.da dict
                xr_data_vars[metric_name] = (
                    ["variable", "lead_time"],
                    metric_rescaled.cpu().numpy().T,
                )

        if (
            self.trainer.is_global_zero
            and not self.trainer.sanity_checking
            and metrics_dict
        ):
            wandb.log(log_dict)  # Log all
            plt.close("all")  # Close all figs

            # Create and save xr.ds
            num_steps = metric_rescaled.shape[0]
            lead_time_i = np.arange(num_steps) + 1  # Lead time in index
            lead_time_h = (
                (self.step_length * lead_time_i)
                .astype("timedelta64[h]")
                .astype("timedelta64[ns]")
            )  # Lead time in hours -> in ns for xr
            metric_ds = xr.Dataset(
                data_vars=xr_data_vars,
                coords={
                    "variable": self._datastore.get_vars_names(
                        category="state"
                    ),
                    "variable_units": (
                        "variable",
                        self._datastore.get_vars_units(category="state"),
                    ),
                    "variable_long_name": (
                        "variable",
                        self._datastore.get_vars_long_names(category="state"),
                    ),
                    "lead_time": lead_time_h,
                },
                attrs={
                    "wandb_run_name": wandb.run.name,
                    "wandb_run_id": wandb.run.id,
                },
            )
            # Save as pickle
            output_path = os.path.join(wandb.run.dir, f"{prefix}_metrics.pkl")
            with open(output_path, "wb") as f:
                pickle.dump(metric_ds, f)

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch. Will
        gather stored tensors and perform plotting and logging on rank 0.
        """
        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.test_metrics, prefix="test")

        # Plot spatial loss maps
        spatial_loss_tensor = self.all_gather_cat(
            torch.cat(self.spatial_loss_maps, dim=0)
        )  # (N_test, N_log, num_interior_nodes)
        if self.trainer.is_global_zero:
            mean_spatial_loss = torch.mean(
                spatial_loss_tensor, dim=0
            )  # (N_log, num_interior_nodes)

            loss_map_figs = [
                vis.plot_spatial_error(
                    error=loss_map,
                    datastore=self._datastore,
                    title=f"Test loss, t={t_i} "
                    f"({self.step_length * t_i} h)",
                )
                for t_i, loss_map in zip(
                    self.args.val_steps_to_log, mean_spatial_loss
                )
            ]

            # log all to same wandb key, sequentially
            for fig in loss_map_figs:
                wandb.log({"test_loss": wandb.Image(fig)})

            # also make without title and save as pdf
            pdf_loss_map_figs = [
                vis.plot_spatial_error(
                    error=loss_map, datastore=self._datastore
                )
                for loss_map in mean_spatial_loss
            ]
            pdf_loss_maps_dir = os.path.join(wandb.run.dir, "spatial_loss_maps")
            os.makedirs(pdf_loss_maps_dir, exist_ok=True)
            for t_i, fig in zip(self.args.val_steps_to_log, pdf_loss_map_figs):
                fig.savefig(os.path.join(pdf_loss_maps_dir, f"loss_t{t_i}.pdf"))
            # save mean spatial loss as .pt file also
            torch.save(
                mean_spatial_loss.cpu(),
                os.path.join(wandb.run.dir, "mean_spatial_loss.pt"),
            )

        self.spatial_loss_maps.clear()

    def on_load_checkpoint(self, checkpoint):
        """
        Perform any changes to state dict before loading checkpoint
        """
        loaded_state_dict = checkpoint["state_dict"]

        # Fix for loading older models after IneractionNet refactoring, where
        # the grid MLP was moved outside the encoder InteractionNet class
        if "g2m_gnn.grid_mlp.0.weight" in loaded_state_dict:
            replace_keys = list(
                filter(
                    lambda key: key.startswith("g2m_gnn.grid_mlp"),
                    loaded_state_dict.keys(),
                )
            )
            for old_key in replace_keys:
                new_key = old_key.replace(
                    "g2m_gnn.grid_mlp", "encoding_grid_mlp"
                )
                loaded_state_dict[new_key] = loaded_state_dict[old_key]
                del loaded_state_dict[old_key]
