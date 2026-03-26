# Third-party
import torch
from torch import nn

# Local
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from ..loss_weighting import get_state_feature_weighting


class StepPredictor(nn.Module):
    """
    Base class for one-step predictors mapping
    (X_{t-1}, X_t, forcing_t) -> X_{t+1}.
    """

    def __init__(
        self,
        args,
        config: NeuralLAMConfig,
        datastore: BaseDatastore,
    ):
        super().__init__()
        self.args = args
        self._datastore = datastore

        # Data dimensions
        num_state_vars = datastore.get_num_data_vars(category="state")
        num_forcing_vars = datastore.get_num_data_vars(category="forcing")
        num_past_forcing_steps = args.num_past_forcing_steps
        num_future_forcing_steps = args.num_future_forcing_steps

        # Static grid features are required by existing graph predictors
        da_static_features = datastore.get_dataarray(
            category="static", split=None, standardize=True
        )
        if da_static_features is None:
            raise ValueError("Static features are required for StepPredictor")

        # Register static features
        self.register_buffer(
            "grid_static_features",
            torch.tensor(da_static_features.values, dtype=torch.float32),
            persistent=False,
        )

        # Output dimensions
        self.output_std = bool(args.output_std)
        if self.output_std:
            self.grid_output_dim = 2 * num_state_vars
        else:
            self.grid_output_dim = num_state_vars

        # Grid dimensions from input state + static + forcing window
        (
            self.num_grid_nodes,
            grid_static_dim,
        ) = self.grid_static_features.shape

        self.grid_dim = (
            2 * num_state_vars
            + grid_static_dim
            + num_forcing_vars
            * (num_past_forcing_steps + num_future_forcing_steps + 1)
        )

        # Standardization statistics
        da_state_stats = datastore.get_standardization_dataarray(category="state")
        state_stats = {
            "state_mean": torch.tensor(
                da_state_stats.state_mean.values, dtype=torch.float32
            ),
            "state_std": torch.tensor(
                da_state_stats.state_std.values, dtype=torch.float32
            ),
            "diff_mean": torch.tensor(
                da_state_stats.state_diff_mean_standardized.values,
                dtype=torch.float32,
            ),
            "diff_std": torch.tensor(
                da_state_stats.state_diff_std_standardized.values,
                dtype=torch.float32,
            ),
        }
        for key, val in state_stats.items():
            self.register_buffer(key, val, persistent=False)

        # Feature weighting is currently still used for constant output std
        state_feature_weights = get_state_feature_weighting(
            config=config, datastore=datastore
        )
        self.register_buffer(
            "feature_weights",
            torch.tensor(state_feature_weights, dtype=torch.float32),
            persistent=False,
        )

        if not self.output_std:
            self.register_buffer(
                "per_var_std",
                self.diff_std / torch.sqrt(self.feature_weights),
                persistent=False,
            )

        # Boundary/interior masks
        da_boundary_mask = datastore.boundary_mask
        boundary_mask = torch.tensor(
            da_boundary_mask.values, dtype=torch.float32
        ).unsqueeze(1)
        self.register_buffer("boundary_mask", boundary_mask, persistent=False)
        self.register_buffer(
            "interior_mask", 1.0 - self.boundary_mask, persistent=False
        )

    @staticmethod
    def expand_to_batch(x, batch_size):
        """
        Expand tensor with initial batch dimension.
        """
        return x.unsqueeze(0).expand(batch_size, -1, -1)

    def predict_step(self, prev_state, prev_prev_state, forcing):
        """
        Predict one step ahead from previous states and forcing.
        """
        raise NotImplementedError("Subclasses must implement predict_step")
