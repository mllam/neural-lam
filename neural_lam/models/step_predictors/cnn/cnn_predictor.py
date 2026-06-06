# Standard library
from typing import Dict, Optional

# Third-party
import torch

# Local
from ....cnn_layers import ResHRRRBackbone, grid_to_node, node_to_grid
from ....datastore.base import BaseDatastore, BaseRegularGridDatastore
from ..base import StepPredictor


class CNNPredictor(StepPredictor):
    """
    ResHRRR-style CNN step predictor for regular-grid datastores.

    The predictor keeps the Neural-LAM ``StepPredictor`` interface in node
    space, but runs the CNN backbone on regular-grid tensors.
    """

    def __init__(
        self,
        datastore: BaseDatastore,
        cnn_channels: int = 128,
        cnn_blocks: int = 8,
        cnn_kernel_size: int = 3,
        cnn_se_reduction: int = 16,
        cnn_film: bool = False,
        cnn_padding_mode: str = "zeros",
        num_past_forcing_steps: int = 1,
        num_future_forcing_steps: int = 1,
        output_std: bool = False,
        output_clamping_lower: Optional[Dict[str, float]] = None,
        output_clamping_upper: Optional[Dict[str, float]] = None,
    ):
        if not isinstance(datastore, BaseRegularGridDatastore):
            raise TypeError("CNNPredictor requires a BaseRegularGridDatastore")

        super().__init__(
            datastore=datastore,
            output_std=output_std,
            output_clamping_lower=output_clamping_lower,
            output_clamping_upper=output_clamping_upper,
        )

        self.grid_shape = datastore.grid_shape_state
        expected_nodes = self.grid_shape.x * self.grid_shape.y
        if expected_nodes != self.num_grid_nodes:
            raise ValueError(
                "datastore grid_shape_state does not match num_grid_points: "
                f"got {expected_nodes}, expected {self.num_grid_nodes}"
            )

        da_state_stats = datastore.get_standardization_dataarray("state")
        self.register_buffer(
            "diff_mean",
            torch.tensor(
                da_state_stats.state_diff_mean_standardized.values,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "diff_std",
            torch.tensor(
                da_state_stats.state_diff_std_standardized.values,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.num_state_vars = datastore.get_num_data_vars(category="state")
        num_forcing_vars = datastore.get_num_data_vars(category="forcing")
        forcing_window_steps = (
            num_past_forcing_steps + num_future_forcing_steps + 1
        )
        self.forcing_input_dim = num_forcing_vars * forcing_window_steps

        grid_static_dim = self.grid_static_features.shape[1]
        self.grid_input_dim = (
            2 * self.num_state_vars + self.forcing_input_dim + grid_static_dim
        )
        context_dim = self.forcing_input_dim if cnn_film else None

        self.backbone = ResHRRRBackbone(
            input_channels=self.grid_input_dim,
            output_channels=self.grid_output_dim,
            hidden_channels=cnn_channels,
            num_blocks=cnn_blocks,
            kernel_size=cnn_kernel_size,
            reduction=cnn_se_reduction,
            context_dim=context_dim,
            padding_mode=cnn_padding_mode,
        )

        self.cnn_film = cnn_film
        self.prepare_clamping_params(datastore)

    def forward(
        self,
        prev_state: torch.Tensor,
        prev_prev_state: torch.Tensor,
        forcing: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._validate_inputs(prev_state, prev_prev_state, forcing)
        batch_size = prev_state.shape[0]

        grid_features = torch.cat(
            (
                prev_state,
                prev_prev_state,
                forcing,
                self.expand_to_batch(self.grid_static_features, batch_size),
            ),
            dim=-1,
        )
        grid_features = node_to_grid(grid_features, self.grid_shape)

        context = forcing.mean(dim=1) if self.cnn_film else None
        net_output = self.backbone(grid_features, context=context)
        net_output = grid_to_node(net_output, self.grid_shape)

        if self.output_std:
            pred_delta_mean, pred_std_raw = net_output.chunk(2, dim=-1)
            pred_std = torch.nn.functional.softplus(pred_std_raw)
        else:
            pred_delta_mean = net_output
            pred_std = None

        rescaled_delta_mean = pred_delta_mean * self.diff_std + self.diff_mean
        new_state = self.get_clamped_new_state(rescaled_delta_mean, prev_state)

        return new_state, pred_std

    def _validate_inputs(
        self,
        prev_state: torch.Tensor,
        prev_prev_state: torch.Tensor,
        forcing: torch.Tensor,
    ) -> None:
        if prev_state.ndim != 3:
            raise ValueError(
                "prev_state must have shape (B, N, d_state), "
                f"got {tuple(prev_state.shape)}"
            )
        if prev_prev_state.shape != prev_state.shape:
            raise ValueError(
                "prev_prev_state must have the same shape as prev_state"
            )
        if forcing.ndim != 3:
            raise ValueError(
                "forcing must have shape (B, N, d_forcing), "
                f"got {tuple(forcing.shape)}"
            )
        if forcing.shape[:2] != prev_state.shape[:2]:
            raise ValueError(
                "forcing must have the same batch and node dimensions as "
                "prev_state"
            )
        if prev_state.shape[1] != self.num_grid_nodes:
            raise ValueError(
                f"state node dimension must be {self.num_grid_nodes}, "
                f"got {prev_state.shape[1]}"
            )
        if prev_state.shape[2] != self.num_state_vars:
            raise ValueError(
                f"state feature dimension must be {self.num_state_vars}, "
                f"got {prev_state.shape[2]}"
            )
        if forcing.shape[2] != self.forcing_input_dim:
            raise ValueError(
                f"forcing feature dimension must be {self.forcing_input_dim}, "
                f"got {forcing.shape[2]}"
            )
