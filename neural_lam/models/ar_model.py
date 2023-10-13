import os
import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt

from neural_lam import utils, vis, constants

class ARModel(pl.LightningModule):
    """
    Generic auto-regressive weather model.
    Abstract class that can be extended.
    """
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.lr = args.lr

        # Some constants useful for sub-classes
        self.batch_static_feature_dim = 1 # Only open water?
        self.grid_forcing_dim = 5*3 # 5 features for 3 time-step window
        self.grid_state_dim = 17

        # Load static features for grid/data
        static_data_dict = utils.load_static_data(args.dataset)
        for static_data_name, static_data_tensor in static_data_dict.items():
            self.register_buffer(static_data_name, static_data_tensor, persistent=False)

        # MSE loss, need to do reduction ourselves to get proper weighting
        if args.loss == "mse":
            self.loss = nn.MSELoss(reduction="none")

            inv_var = self.step_diff_std**-2.
            state_weight = self.param_weights*inv_var # (d_f,)
        elif args.loss == "mae":
            self.loss = nn.L1Loss(reduction="none")

            # Weight states with inverse std instead in this case
            state_weight = self.param_weights/self.step_diff_std # (d_f,)
        else:
            assert False, f"Unknown loss function: {args.loss}"
        self.register_buffer("state_weight", state_weight, persistent=False)

        # Pre-compute interior mask for use in loss function
        self.register_buffer("interior_mask", 1. - self.border_mask,
                persistent=False) # (N_grid, 1), 1 for non-border
        self.N_interior = torch.sum(self.interior_mask
                ).item() # Number of grid nodes to predict

        self.step_length = args.step_length # Number of hours per pred. step
        self.val_maes = []
        self.test_maes = []
        self.test_mses = []

        # For making restoring of optimizer state optional (slight hack)
        self.opt_state = None

        # For example plotting
        self.n_example_pred = args.n_example_pred
        self.plotted_examples = 0

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps = []

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.95))
        if self.opt_state:
            opt.load_state_dict(self.opt_state)

        return opt

    @staticmethod
    def expand_to_batch(x, batch_size):
        """
        Expand tensor with initial batch dimension
        """
        return x.unsqueeze(0).expand(batch_size, -1, -1)

    def predict_step(self, prev_state, prev_prev_state, batch_static_features, forcing):
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, N_grid, feature_dim), X_t
        prev_prev_state: (B, N_grid, feature_dim), X_{t-1}
        batch_static_features: (B, N_grid, batch_static_feature_dim)
        forcing: (B, N_grid, forcing_dim)
        """
        raise NotImplementedError("No prediction step implemented")

    def unroll_prediction(self, init_states, batch_static_features, forcing_features,
            true_states):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, 2, N_grid, d_f)
        batch_static_features: (B, N_grid, d_static_f)
        forcing_features: (B, pred_steps, N_grid, d_static_f)
        true_states: (B, pred_steps, N_grid, d_f)
        """
        prev_prev_state = init_states[:,0]
        prev_state = init_states[:,1]
        prediction_list = []
        pred_steps = forcing_features.shape[1]

        for i in range(pred_steps):
            forcing = forcing_features[:, i]
            border_state = true_states[:, i]

            predicted_state = self.predict_step(prev_state, prev_prev_state,
                    batch_static_features, forcing) # (B, N_grid, d_f)

            # Overwrite border with true state
            new_state = self.border_mask*border_state +\
                    self.interior_mask*predicted_state
            prediction_list.append(new_state)

            # Upate conditioning states
            prev_prev_state = prev_state
            prev_state = new_state

        return torch.stack(prediction_list, dim=1) # (B, pred_steps, N_grid, d_f)

    def weighted_loss(self, prediction, target, reduce_spatial_dim=True):
        """
        Computed weighted loss function.
        prediction/target: (B, pred_steps, N_grid, d_f)
        returns (B, pred_steps)
        """
        entry_loss = self.loss(prediction, target) # (B, pred_steps, N_grid, d_f)
        grid_node_loss = torch.sum(entry_loss*self.state_weight,
                dim=-1) # (B, pred_steps, N_grid), weighted sum over features
        if not reduce_spatial_dim:
            return grid_node_loss # (B, pred_steps, N_grid)

        # Take (unweighted) mean over only non-border (interior) grid nodes
        time_step_loss = torch.sum(grid_node_loss*self.interior_mask[:,0],
                dim=-1)/self.N_interior # (B, pred_steps)

        return time_step_loss # (B, pred_steps)

    def common_step(self, batch):
        """
        Predict on single batch
        batch = time_series, batch_static_features, forcing_features

        init_states: (B, 2, N_grid, d_features)
        target_states: (B, pred_steps, N_grid, d_features)
        batch_static_features: (B, N_grid, d_static_f), for example open water
        forcing_features: (B, pred_steps, N_grid, d_forcing), where index 0
            corresponds to index 1 of init_states
        """
        init_states, target_states, batch_static_features, forcing_features = batch

        prediction = self.unroll_prediction(init_states, batch_static_features,
                forcing_features, target_states) # (B, pred_steps, N_grid, d_f)

        return prediction, target_states

    def training_step(self, batch):
        """
        Train on single batch
        """
        prediction, target = self.common_step(batch)

        # Compute loss
        batch_loss = torch.mean(self.weighted_loss(
            prediction, target)) # mean over unrolled times and batch

        log_dict = {"train_loss": batch_loss}
        self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=True,
                sync_dist=True)
        return batch_loss

    def per_var_error(self, prediction, target, error="mae"):
        """
        Computed MAE/MSE per variable and time step
        prediction/target: (B, pred_steps, N_grid, d_f)
        returns (B, pred_steps)
        """
        if error == "mse":
            loss_func = torch.nn.functional.mse_loss
        else:
            loss_func = torch.nn.functional.l1_loss
        entry_loss = loss_func(prediction, target,
                reduction="none") # (B, pred_steps, N_grid, d_f)

        mean_error = torch.sum(entry_loss*self.interior_mask,
                dim=2)/self.N_interior # (B, pred_steps, d_f)
        return mean_error

    def all_gather_cat(self, tensor_to_gather):
        """
        Gather tensors across all ranks, and concatenate across dim. 0 (instead of
        stacking in new dim. 0)

        tensor_to_gather: (d1, d2, ...), distributed over K ranks

        returns: (K*d1, d2, ...)
        """
        return self.all_gather(tensor_to_gather).flatten(0, 1)

    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch
        """
        prediction, target = self.common_step(batch)

        time_step_loss = torch.mean(self.weighted_loss(prediction,
            target), dim=0) # (time_steps-1)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        val_log_dict = {f"val_loss_unroll{step}": time_step_loss[step-1]
                for step in constants.val_step_log_errors}
        val_log_dict["val_mean_loss"] = mean_loss

        maes = self.per_var_error(prediction, target) # (B, pred_steps, d_f)
        self.val_maes.append(maes)

        self.log_dict(val_log_dict, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """
        val_mae_tensor = self.all_gather_cat(torch.cat(
            self.val_maes, dim=0)) # (N_val, pred_steps, d_f)

        if self.trainer.is_global_zero:
            val_mae_total = torch.mean(val_mae_tensor, dim=0) # (pred_steps, d_f)
            val_mae_rescaled = val_mae_total * self.data_std # (pred_steps, d_f)

            if not self.trainer.sanity_checking:
                # Don't log this during sanity checking
                mae_fig = vis.plot_error_map(val_mae_rescaled, title="Validation MAE",
                        step_length=self.step_length)
                wandb.log({"val_mae": wandb.Image(mae_fig)})
                plt.close("all") # Close all figs

        self.val_maes.clear() # Free memory

    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        """
        prediction, target = self.common_step(batch)

        time_step_loss = torch.mean(self.weighted_loss(prediction,
            target), dim=0) # (time_steps-1)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        test_log_dict = {f"test_loss_unroll{step}": time_step_loss[step-1]
                for step in constants.val_step_log_errors}
        test_log_dict["test_mean_loss"] = mean_loss

        self.log_dict(test_log_dict, on_step=False, on_epoch=True, sync_dist=True)

        # For error maps
        maes = self.per_var_error(prediction, target) # (B, pred_steps, d_f)
        self.test_maes.append(maes)
        mses = self.per_var_error(prediction, target, error="mse") # (B, pred_steps, d_f)
        self.test_mses.append(mses)

        # Save per-sample spatial loss for specific times
        spatial_loss = self.weighted_loss(prediction, target,
                reduce_spatial_dim=False) # (B, pred_steps, N_grid)
        log_spatial_losses = spatial_loss[:,constants.val_step_log_errors-1]
        self.spatial_loss_maps.append(log_spatial_losses) # (B, N_log, N_grid)

        # Plot example predictions (on rank 0 only)
        if self.trainer.is_global_zero and self.plotted_examples < self.n_example_pred:
            # Need to plot more example predictions
            n_additional_examples = min(prediction.shape[0], self.n_example_pred
                    - self.plotted_examples)

            # Rescale to original data scale
            prediction_rescaled = prediction*self.data_std + self.data_mean
            target_rescaled = target*self.data_std + self.data_mean

            # Iterate over the examples
            for pred_slice, target_slice in zip(
                    prediction_rescaled[:n_additional_examples],
                    target_rescaled[:n_additional_examples]):
                # Each slice is (pred_steps, N_grid, d_f)
                self.plotted_examples += 1 # Increment already here

                var_vmin = torch.minimum(
                        pred_slice.flatten(0,1).min(dim=0)[0],
                        target_slice.flatten(0,1).min(dim=0)[0]).cpu().numpy() # (d_f,)
                var_vmax = torch.maximum(
                        pred_slice.flatten(0,1).max(dim=0)[0],
                        target_slice.flatten(0,1).max(dim=0)[0]).cpu().numpy() # (d_f,)
                var_vranges = list(zip(var_vmin, var_vmax))

                # Iterate over prediction horizon time steps
                for t_i, (pred_t, target_t) in enumerate(zip(pred_slice, target_slice),
                        start=1):
                    # Create one figure per variable at this time step
                    var_figs = [vis.plot_prediction(
                        pred_t[:,var_i], target_t[:,var_i],
                        self.interior_mask[:,0],
                        title=f"{var_name} ({var_unit}), "
                            f"t={t_i} ({self.step_length*t_i} h)",
                        vrange=var_vrange)
                            for var_i, (var_name, var_unit, var_vrange) in
                            enumerate(zip(
                                constants.param_names_short,
                                constants.param_units,
                                var_vranges
                            ))
                        ]

                    wandb.log({
                            f"{var_name}_example_{self.plotted_examples}":
                                wandb.Image(fig)
                        for var_name, fig in zip(constants.param_names_short, var_figs)
                    })
                    plt.close("all") # Close all figs for this time step, saves memory

                # Save pred and target as .pt files
                torch.save(pred_slice.cpu(),os.path.join(
                    wandb.run.dir, f'example_pred_{self.plotted_examples}.pt'))
                torch.save(target_slice.cpu(),os.path.join(
                    wandb.run.dir, f'example_target_{self.plotted_examples}.pt'))

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch.
        Will gather stored tensors and perform plotting and logging on rank 0.
        """
        # Create error maps for RMSE and MAE
        test_mae_tensor = self.all_gather_cat(torch.cat(self.test_maes,
            dim=0)) # (N_test, pred_steps, d_f)
        test_mse_tensor = self.all_gather_cat(torch.cat(self.test_mses,
            dim=0)) # (N_test, pred_steps, d_f)

        if self.trainer.is_global_zero:
            test_mae_rescaled = torch.mean(test_mae_tensor,
                    dim=0) * self.data_std # (pred_steps, d_f)
            test_rmse_rescaled = torch.sqrt(torch.mean(test_mae_tensor,
                    dim=0)) * self.data_std # (pred_steps, d_f)

            mae_fig = vis.plot_error_map(test_mae_rescaled, step_length=self.step_length)
            rmse_fig = vis.plot_error_map(test_rmse_rescaled, step_length=self.step_length)
            wandb.log({ # Log png:s
                "test_mae": wandb.Image(mae_fig),
                "test_rmse": wandb.Image(rmse_fig),
                })
            # Save pdf:s
            mae_fig.savefig(os.path.join(wandb.run.dir, "test_mae.pdf"))
            rmse_fig.savefig(os.path.join(wandb.run.dir, "test_rmse.pdf"))
            # Save errors also as csv:s
            np.savetxt(os.path.join(wandb.run.dir, "test_mae.csv"),
                    test_mae_rescaled.cpu().numpy(), delimiter=",")
            np.savetxt(os.path.join(wandb.run.dir, "test_rmse.csv"),
                    test_rmse_rescaled.cpu().numpy(), delimiter=",")

        self.test_maes.clear() # Free memory
        self.test_mses.clear()

        # Plot spatial loss maps
        spatial_loss_tensor = self.all_gather_cat(torch.cat(self.spatial_loss_maps,
                dim=0)) # (N_test, N_log, N_grid)
        if self.trainer.is_global_zero:
            mean_spatial_loss = torch.mean(spatial_loss_tensor, dim=0) # (N_log, N_grid)

            loss_map_figs = [vis.plot_spatial_error(loss_map, self.interior_mask[:,0],
                    title=f"Test loss, t={t_i} ({self.step_length*t_i} h)")
                for t_i, loss_map in zip(constants.val_step_log_errors, mean_spatial_loss)]

            # log all to same wandb key, sequentially
            for fig in loss_map_figs:
                wandb.log({"test_loss": wandb.Image(fig)})

            # also make without title and save as pdf
            pdf_loss_map_figs = [vis.plot_spatial_error(loss_map, self.interior_mask[:,0])
                for loss_map in mean_spatial_loss]
            pdf_loss_maps_dir = os.path.join(wandb.run.dir, "spatial_loss_maps")
            os.makedirs(pdf_loss_maps_dir, exist_ok=True)
            for t_i, fig in zip(constants.val_step_log_errors,pdf_loss_map_figs):
                fig.savefig(os.path.join(pdf_loss_maps_dir, f"loss_t{t_i}.pdf"))
            # save mean spatial loss as .pt file also
            torch.save(mean_spatial_loss.cpu(),os.path.join(
                wandb.run.dir, 'mean_spatial_loss.pt'))

        self.spatial_loss_maps.clear()
