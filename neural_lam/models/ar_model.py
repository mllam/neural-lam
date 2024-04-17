# pylint: disable=wrong-import-order
# Standard library
import glob
import os
from datetime import datetime, timedelta

# Third-party
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.utilities import rank_zero_only
from torch import nn

# First-party
from neural_lam import constants, metrics, utils, vis


# pylint: disable=too-many-public-methods
class ARModel(pl.LightningModule):
    """
    Generic auto-regressive weather model.
    Abstract class that can be extended.
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()
        self.lr = args.lr

        # Log prediction error for these time steps forward
        self.val_step_log_errors = constants.VAL_STEP_LOG_ERRORS
        self.metrics_initialized = constants.METRICS_INITIALIZED

        # Some constants useful for sub-classes
        self.grid_forcing_dim = constants.GRID_FORCING_DIM
        count_3d_fields = sum(value == 1 for value in constants.IS_3D.values())
        count_2d_fields = sum(value != 1 for value in constants.IS_3D.values())
        self.grid_state_dim = (
            len(constants.VERTICAL_LEVELS) * count_3d_fields + count_2d_fields
        )

        # Load static features for grid/data
        static_data_dict = utils.load_static_data(args.dataset)
        for static_data_name, static_data_tensor in static_data_dict.items():
            self.register_buffer(
                static_data_name, static_data_tensor, persistent=False
            )

        # Double grid output dim. to also output std.-dev.
        self.output_std = bool(args.output_std)
        if self.output_std:
            self.grid_output_dim = (
                2 * constants.GRID_STATE_DIM
            )  # Pred. dim. in grid cell
        else:
            self.grid_output_dim = (
                constants.GRID_STATE_DIM
            )  # Pred. dim. in grid cell

            # Store constant per-variable std.-dev. weighting
            # Note that this is the inverse of the multiplicative weighting
            # in wMSE/wMAE
            self.register_buffer(
                "per_var_std",
                self.step_diff_std / torch.sqrt(self.param_weights),
                persistent=False,
            )

        # grid_dim from data + static
        (
            self.num_grid_nodes,
            grid_static_dim,
        ) = self.grid_static_features.shape  # 63784 = 268x238
        self.grid_dim = (
            2 * constants.GRID_STATE_DIM
            + grid_static_dim
            + constants.GRID_FORCING_DIM
        )

        # Instantiate loss function
        self.loss = metrics.get_metric(args.loss)

        # Pre-compute interior mask for use in loss function
        self.register_buffer(
            "interior_mask", 1.0 - self.border_mask, persistent=False
        )  # (num_grid_nodes, 1), 1 for non-border

        self.step_length = args.step_length  # Number of hours per pred. step
        self.val_metrics = {
            "mse": [],
        }
        self.test_metrics = {
            "mse": [],
            "mae": [],
        }
        if self.output_std:
            self.test_metrics["output_std"] = []  # Treat as metric

        # For making restoring of optimizer state optional
        self.restore_opt = args.restore_opt

        # For example plotting
        self.n_example_pred = args.n_example_pred

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps = []
        # For storing prediction output
        self.inference_output = []

        self.variable_indices = self.precompute_variable_indices()
        self.selected_vars_units = [
            (var_name, var_unit)
            for var_name, var_unit in zip(
                constants.PARAM_NAMES_SHORT, constants.PARAM_UNITS
            )
            if var_name in constants.EVAL_PLOT_VARS
        ]

        utils.rank_zero_print("variable_indices", self.variable_indices)
        utils.rank_zero_print("selected_vars_units", self.selected_vars_units)

    @pl.utilities.rank_zero_only
    def log_image(self, name, img):
        """Log an image to wandb"""
        wandb.log({name: wandb.Image(img)})

    @pl.utilities.rank_zero_only
    def init_metrics(self):
        """
        Set up wandb metrics to track
        """

        wandb.define_metric("val_mean_loss", summary="min")
        for step in self.val_step_log_errors:
            wandb.define_metric(f"val_loss_unroll{step:02}", summary="min")
        self.metrics_initialized = True  # Make sure this is done only once

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, betas=(0.9, 0.95)
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt, step_size=30, gamma=0.1
        )

        return [opt], [scheduler]

    @property
    def interior_mask_bool(self):
        """
        Get the interior mask as a boolean (N,) mask.
        """
        return self.interior_mask[:, 0].to(torch.bool)

    @staticmethod
    def expand_to_batch(x, batch_size):
        """
        Expand tensor with initial batch dimension
        """
        return x.unsqueeze(0).expand(batch_size, -1, -1)

    def precompute_variable_indices(self):
        """
        Precompute indices for each variable in the input tensor
        """
        variable_indices = {}
        all_vars = []
        index = 0
        # Create a list of tuples for all variables, using level 0 for 2D
        # variables
        for var_name in constants.PARAM_NAMES_SHORT:
            if constants.IS_3D[var_name]:
                for level in constants.VERTICAL_LEVELS:
                    all_vars.append((var_name, level))
            else:
                all_vars.append((var_name, 0))  # Use level 0 for 2D variables

        # Sort the variables based on the tuples
        sorted_vars = sorted(all_vars)

        for var in sorted_vars:
            var_name, level = var
            if var_name not in variable_indices:
                variable_indices[var_name] = []
            variable_indices[var_name].append(index)
            index += 1

        return variable_indices

    def apply_constraints(self, prediction):
        """
        Apply constraints to prediction to ensure values are within the
        specified bounds
        """
        for param, (min_val, max_val) in constants.PARAM_CONSTRAINTS.items():
            indices = self.variable_indices[param]
            for index in indices:
                # Apply clamping to ensure values are within the specified
                # bounds
                prediction[:, :, index] = torch.clamp(
                    prediction[:, :, index],
                    min=min_val,
                    max=max_val if max_val is not None else float("inf"),
                )
        return prediction

    def single_prediction(
        self,
        prev_state,
        prev_prev_state,
        forcing,
    ):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t
        prev_prev_state: (B, num_grid_nodes, feature_dim), X_{t-1}
        forcing: (B, num_grid_nodes, forcing_dim)
        """
        raise NotImplementedError("No prediction step implemented")

    # pylint: disable-next=unused-argument
    def predict_step(self, batch, batch_idx):
        """
        Run the inference on batch.
        """
        prediction, target, pred_std, _ = self.common_step(batch)

        # Compute all evaluation metrics for error maps
        # Note: explicitly list metrics here, as test_metrics can contain
        # additional ones, computed differently, but that should be aggregated
        # on_predict_epoch_end
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
        log_spatial_losses = spatial_loss[:, constants.VAL_STEP_LOG_ERRORS - 1]
        self.spatial_loss_maps.append(log_spatial_losses)
        # (B, N_log, num_grid_nodes)

        if self.trainer.global_rank == 0:
            self.plot_examples(batch, prediction=prediction)
        self.inference_output.append(prediction)

    def unroll_prediction(self, init_states, forcing_features, true_states):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, 2, num_grid_nodes, d_f)
        forcing_features: (B, pred_steps, num_grid_nodes, d_static_f)
        true_states: (B, pred_steps, num_grid_nodes, d_f)
        """
        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []
        pred_steps = (
            forcing_features.shape[1]
            if forcing_features is not None
            else true_states.shape[1]
        )

        for i in range(pred_steps):
            forcing = (
                forcing_features[:, i] if forcing_features is not None else None
            )
            border_state = true_states[:, i]

            pred_state, pred_std = self.single_prediction(
                prev_state, prev_prev_state, forcing
            )
            # state: (B, num_grid_nodes, d_f)
            # pred_std: (B, num_grid_nodes, d_f) or None

            # Overwrite border with true state
            new_state = (
                self.border_mask * border_state
                + self.interior_mask * pred_state
            )

            prediction_list.append(new_state)
            if self.output_std:
                pred_std_list.append(pred_std)

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = new_state

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        if self.output_std:
            pred_std = torch.stack(
                pred_std_list, dim=1
            )  # (B, pred_steps, num_grid_nodes, d_f)
        else:
            pred_std = self.per_var_std  # (d_f,)

        return prediction, pred_std

    def common_step(self, batch):
        """
        Predict on single batch
        batch consists of:
        init_states: (B, 2, num_grid_nodes, d_features)
        target_states: (B, pred_steps, num_grid_nodes, d_features)
        forcing_features: (B, pred_steps, num_grid_nodes, d_forcing),
            where index 0 corresponds to index 1 of init_states
        """
        init_states, target_states, batch_time = batch[:3]
        forcing_features = batch[4] if len(batch) > 3 else None

        prediction, pred_std = self.unroll_prediction(
            init_states, forcing_features, target_states
        )  # (B, pred_steps, num_grid_nodes, d_f)
        # prediction: (B, pred_steps, num_grid_nodes, d_f)
        # pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)

        return prediction, target_states, pred_std, batch_time

    def training_step(self, batch):
        """
        Train on single batch
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
        Gather tensors across all ranks, and concatenate across dim. 0
        (instead of stacking in new dim. 0)

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
                prediction, target, pred_std, mask=self.interior_mask_bool
            ),
            dim=0,
        )  # (time_steps-1)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
            for step in constants.VAL_STEP_LOG_ERRORS
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
        Compute val metrics at the end of val epoch
        """
        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.val_metrics, prefix="val")

        # Clear lists with validation metrics values
        for metric_list in self.val_metrics.values():
            metric_list.clear()

    # pylint: disable-next=unused-argument
    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        """
        prediction, target, pred_std, batch_time = self.common_step(batch)
        # prediction: (B, pred_steps, num_grid_nodes, d_f)
        # pred_std: (B, pred_steps, num_grid_nodes, d_f) or (d_f,)

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
            for step in constants.VAL_STEP_LOG_ERRORS
        }
        test_log_dict["test_mean_loss"] = mean_loss

        self.log_dict(
            test_log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )

        # Compute all evaluation metrics for error maps
        # Note: explicitly list metrics here, as test_metrics can contain
        # additional ones, computed differently, but that should be aggregated
        # on_test_epoch_end
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
        log_spatial_losses = spatial_loss[:, constants.VAL_STEP_LOG_ERRORS - 1]
        self.spatial_loss_maps.append(log_spatial_losses)
        # (B, N_log, num_grid_nodes)

        if self.trainer.is_global_zero:
            self.plot_examples(
                batch,
                prediction=prediction,
                target=target,
                batch_time=batch_time,
            )

    @rank_zero_only
    def plot_examples(
        self, batch, prediction=None, target=None, batch_time=None
    ):
        """
        Plot the first n_examples forecasts from batch

        Parameters:
        - batch: batch with data to plot corresponding forecasts for
        - n_examples: number of forecasts to plot
        - prediction: (B, pred_steps, num_grid_nodes, d_f), existing prediction.
            Generate if None.

        The function checks for the presence of test_dataset or
        predict_dataset within the trainer's data module,
        handles indexing within the batch for targeted analysis,
        performs prediction rescaling, and plots results.
        """
        if prediction is None or target is None or batch_time is None:
            prediction, target, _, batch_time = self.common_step(batch)

        if self.global_rank == 0 and any(
            eval_datetime in batch_time
            for eval_datetime in constants.EVAL_DATETIMES
        ):
            # Rescale to original data scale
            prediction_rescaled = prediction * self.data_std + self.data_mean
            prediction_rescaled = self.apply_constraints(prediction_rescaled)
            target_rescaled = target * self.data_std + self.data_mean

            if constants.SMOOTH_BOUNDARIES:
                prediction_rescaled = self.smooth_prediction_borders(
                    prediction_rescaled
                )

            for i, eval_datetime in enumerate(batch_time):
                if eval_datetime not in constants.EVAL_DATETIMES:
                    continue
                pred_rescaled = prediction_rescaled[i]
                targ_rescaled = target_rescaled[i]

                for var_name, var_unit in self.selected_vars_units:
                    var_indices = self.variable_indices[var_name]
                    for lvl_i, var_i in enumerate(var_indices):
                        lvl = constants.VERTICAL_LEVELS[lvl_i]
                        var_vmin = min(
                            pred_rescaled[:, var_i].min(),
                            targ_rescaled[:, var_i].min(),
                        )
                        var_vmax = max(
                            pred_rescaled[:, var_i].max(),
                            targ_rescaled[:, var_i].max(),
                        )
                        var_vrange = (var_vmin, var_vmax)

                        for t_i, (pred_t, target_t) in enumerate(
                            zip(pred_rescaled, targ_rescaled), start=1
                        ):
                            print(f"Plotting {var_name} lvl {lvl_i} t {t_i}...")
                            current_datetime_str = (
                                datetime.strptime(eval_datetime, "%Y%m%d%H")
                                + timedelta(hours=t_i)
                            ).strftime("%Y%m%d%H")
                            title = (
                                f"{var_name} ({var_unit}), "
                                f"t={current_datetime_str}"
                            )
                            var_fig = vis.plot_prediction(
                                pred_t[:, var_i],
                                target_t[:, var_i],
                                title=title,
                                vrange=var_vrange,
                            )
                            wandb.log(
                                {
                                    f"{var_name}_lvl_{lvl:02}_t_"
                                    f"{current_datetime_str}": wandb.Image(
                                        var_fig
                                    )
                                }
                            )
                            plt.close("all")

                if constants.STORE_EXAMPLE_DATA:
                    torch.save(
                        pred_rescaled.cpu(),
                        os.path.join(
                            wandb.run.dir, f"example_pred_{eval_datetime}.pt"
                        ),
                    )
                    torch.save(
                        targ_rescaled.cpu(),
                        os.path.join(
                            wandb.run.dir, f"example_target_{eval_datetime}.pt"
                        ),
                    )

    @rank_zero_only
    def smooth_prediction_borders(self, prediction_rescaled):
        """
        Smooths the prediction at the borders to avoid artifacts.

        Args:
            prediction_rescaled (torch.Tensor): The rescaled prediction tensor.

        Returns:
            torch.Tensor: The prediction tensor after smoothing the borders.
        """
        height, width = constants.GRID_SHAPE
        prediction_permuted = prediction_rescaled.permute(0, 2, 1).reshape(
            prediction_rescaled.size(0),
            prediction_rescaled.size(2),
            height,
            width,
        )

        # Define the smoothing kernel for grouped convolution
        num_groups = prediction_permuted.shape[1]
        kernel_size = 3
        kernel = torch.ones((num_groups, 1, kernel_size, kernel_size)) / (
            kernel_size**2
        )
        kernel = kernel.to(self.device)

        # Use the updated kernel in the conv2d operation
        # pylint: disable-next=not-callable
        prediction_smoothed = nn.functional.conv2d(
            prediction_permuted, kernel, padding=1, groups=num_groups
        )

        # Combine the height and width dimensions back into a single N_grid
        # dimension
        prediction_smoothed = prediction_smoothed.reshape(
            prediction_smoothed.size(0), prediction_smoothed.size(1), -1
        )

        # Permute the dimensions to get back to the original order
        prediction_smoothed = prediction_smoothed.permute(0, 2, 1)

        # Apply the mask to the smoothed prediction
        prediction_rescaled = (
            self.border_mask * prediction_smoothed
            + self.interior_mask * prediction_rescaled
        )

        return prediction_rescaled

    def create_metric_log_dict(self, metric_tensor, prefix, metric_name):
        """
        Put together a dict with everything to log for one metric.
        Also saves plots as pdf and csv if using test prefix.

        metric_tensor: (pred_steps, d_f), metric values per time and variable
        prefix: string, prefix to use for logging
        metric_name: string, name of the metric

        Return:
        log_dict: dict with everything to log for given metric
        """
        log_dict = {}
        metric_fig = vis.plot_error_map(
            metric_tensor, self.data_mean, step_length=self.step_length
        )
        full_log_name = f"{prefix}_{metric_name}"
        log_dict[full_log_name] = wandb.Image(metric_fig)

        if prefix == "test":
            # Save pdf
            metric_fig.savefig(
                os.path.join(wandb.run.dir, f"{full_log_name}.pdf")
            )
            # Save errors also as csv
            np.savetxt(
                os.path.join(wandb.run.dir, f"{full_log_name}.csv"),
                metric_tensor.cpu().numpy(),
                delimiter=",",
            )

        # Check if metrics are watched, log exact values for specific vars
        if full_log_name in constants.METRICS_WATCH:
            for var_i, timesteps in constants.VAR_LEADS_METRICS_WATCH.items():
                var = constants.PARAM_NAMES_SHORT[var_i]
                log_dict.update(
                    {
                        f"{full_log_name}_{var}_step_{step}": metric_tensor[
                            step - 1, var_i
                        ]  # 1-indexed in constants
                        for step in timesteps
                    }
                )

        return log_dict

    def aggregate_and_plot_metrics(self, metrics_dict, prefix):
        """
        Aggregate and create error map plots for all metrics in metrics_dict

        metrics_dict: dictionary with metric_names and list of tensors
            with step-evals.
        prefix: string, prefix to use for logging
        """
        log_dict = {}
        for metric_name, metric_val_list in metrics_dict.items():
            metric_tensor = torch.cat(metric_val_list, dim=0)

            if self.trainer.is_global_zero:
                metric_tensor_averaged = torch.mean(metric_tensor, dim=0)
                # (pred_steps, d_f)

                # Take square root after all averaging to change MSE to RMSE
                if "mse" in metric_name:
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name.replace("mse", "rmse")

                # Note: we here assume rescaling for all metrics is linear
                metric_rescaled = metric_tensor_averaged * self.data_std
                # (pred_steps, d_f)
                log_dict.update(
                    self.create_metric_log_dict(
                        metric_rescaled, prefix, metric_name
                    )
                )

        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            wandb.log(log_dict)  # Log all
            plt.close("all")  # Close all figs

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch.
        Will gather stored tensors and perform plotting and logging on rank 0.
        """
        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.test_metrics, prefix="test")

        # Plot spatial loss maps
        spatial_loss_tensor = self.all_gather_cat(
            torch.cat(self.spatial_loss_maps, dim=0)
        )  # (N_test, N_log, num_grid_nodes)
        if self.trainer.is_global_zero:
            mean_spatial_loss = torch.mean(
                spatial_loss_tensor, dim=0
            )  # (N_log, num_grid_nodes)

            loss_map_figs = [
                vis.plot_spatial_error(
                    loss_map,
                    title=f"Test loss, t={t_i} ({self.step_length * t_i} h)",
                )
                for t_i, loss_map in zip(
                    constants.VAL_STEP_LOG_ERRORS, mean_spatial_loss
                )
            ]

            # log all to same wandb key, sequentially
            for fig in loss_map_figs:
                wandb.log({"test_loss": wandb.Image(fig)})

            # also make without title and save as pdf
            pdf_loss_map_figs = [
                vis.plot_spatial_error(loss_map)
                for loss_map in mean_spatial_loss
            ]
            pdf_loss_maps_dir = os.path.join(wandb.run.dir, "spatial_loss_maps")
            os.makedirs(pdf_loss_maps_dir, exist_ok=True)
            for t_i, fig in zip(
                constants.VAL_STEP_LOG_ERRORS, pdf_loss_map_figs
            ):
                fig.savefig(os.path.join(pdf_loss_maps_dir, f"loss_t{t_i}.pdf"))
            # save mean spatial loss as .pt file also
            torch.save(
                mean_spatial_loss.cpu(),
                os.path.join(wandb.run.dir, "mean_spatial_loss.pt"),
            )

            dir_path = f"{wandb.run.dir}/media/images"

            for var_name, _ in self.selected_vars_units:
                var_indices = self.variable_indices[var_name]
                for lvl_i, _ in enumerate(var_indices):
                    # Calculate var_vrange for each index
                    lvl = constants.VERTICAL_LEVELS[lvl_i]

                    # Get all the images for the current variable and index
                    images = sorted(
                        glob.glob(
                            f"{dir_path}/{var_name}_test_lvl_{lvl:02}_t_*.png"
                        )
                    )
                    # Generate the GIF
                    with imageio.get_writer(
                        f"{dir_path}/{var_name}_lvl_{lvl:02}.gif",
                        mode="I",
                        fps=1,
                    ) as writer:
                        for filename in images:
                            image = imageio.imread(filename)
                            writer.append_data(image)
        self.spatial_loss_maps.clear()

    @rank_zero_only
    def on_predict_epoch_end(self):
        """
        Return inference plot at the end of predict epoch.
        """
        plot_dir_path = f"{wandb.run.dir}/media/images"
        value_dir_path = f"{wandb.run.dir}/results/inference"
        # Ensure the directory for saving numpy arrays exists
        os.makedirs(plot_dir_path, exist_ok=True)
        os.makedirs(value_dir_path, exist_ok=True)

        # For values
        for i, prediction in enumerate(self.inference_output):

            # Rescale to original data scale
            prediction_rescaled = prediction * self.data_std + self.data_mean
            prediction_rescaled = self.apply_constraints(prediction_rescaled)

            if constants.SMOOTH_BOUNDARIES:
                prediction_rescaled = self.smooth_prediction_borders(
                    prediction_rescaled
                )

            # Process and save the prediction
            prediction_array = prediction_rescaled.cpu().numpy()
            file_path = os.path.join(value_dir_path, f"prediction_{i}.npy")
            np.save(file_path, prediction_array)

        # For plots
        for var_name, _ in self.selected_vars_units:
            var_indices = self.variable_indices[var_name]
            for lvl_i, _ in enumerate(var_indices):
                # Calculate var_vrange for each index
                lvl = constants.VERTICAL_LEVELS[lvl_i]

                # Get all the images for the current variable and index
                images = sorted(
                    glob.glob(
                        f"{plot_dir_path}/"
                        f"{var_name}_prediction_lvl_{lvl:02}_t_*.png"
                    )
                )
                # Generate the GIF
                with imageio.get_writer(
                    f"{plot_dir_path}/{var_name}_prediction_lvl_{lvl:02}.gif",
                    mode="I",
                    fps=1,
                ) as writer:
                    for filename in images:
                        image = imageio.imread(filename)
                        writer.append_data(image)

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
        if not self.restore_opt:
            # Create new optimizer and scheduler instances instead of setting
            # them to None
            optimizers, lr_schedulers = self.configure_optimizers()
            checkpoint["optimizer_states"] = [
                opt.state_dict() for opt in optimizers
            ]
            checkpoint["lr_schedulers"] = [
                sched.state_dict() for sched in lr_schedulers
            ]
