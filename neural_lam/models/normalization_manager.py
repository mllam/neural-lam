# Standard library
from typing import Dict

# Third-party
import torch
import torch.nn as nn

# First-party
from ..datastore import BaseDatastore


class NormalizationManager(nn.Module):
    """
    Manages state normalization and statistical buffers for weather models.

    This class centralizes the handling of normalization statistics including
    state mean/std and difference mean/std, as well as static grid features.
    All statistics are registered as buffers to ensure proper device handling
    and state dict serialization.
    """

    def __init__(self, datastore: BaseDatastore):
        """
        Initialize the NormalizationManager.

        Parameters
        ----------
        datastore : BaseDatastore
            The datastore containing the data statistics and static features.
        """
        super().__init__()

        # Load static features standardized
        da_static_features = datastore.get_dataarray(
            category="static", split=None, standardize=True
        )
        if da_static_features is None:
            raise ValueError("Static features are required for NormalizationManager")

        da_state_stats = datastore.get_standardization_dataarray(category="state")

        # Load static features for grid/data
        self.register_buffer(
            "grid_static_features",
            torch.tensor(da_static_features.values, dtype=torch.float32),
            persistent=False,
        )

        # Register state statistics as buffers
        state_stats = {
            "state_mean": torch.tensor(
                da_state_stats.state_mean.values, dtype=torch.float32
            ),
            "state_std": torch.tensor(
                da_state_stats.state_std.values, dtype=torch.float32
            ),
            # Note that the one-step-diff stats (diff_mean and diff_std) are
            # for differences computed on standardized data
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

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Normalize state using mean and standard deviation.

        Parameters
        ----------
        state : torch.Tensor
            The state tensor to normalize, shape (..., d_f)

        Returns
        -------
        torch.Tensor
            Normalized state with same shape as input
        """
        return (state - self.state_mean) / self.state_std

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized state back to original scale.

        Parameters
        ----------
        normalized_state : torch.Tensor
            The normalized state tensor, shape (..., d_f)

        Returns
        -------
        torch.Tensor
            State in original scale with same shape as input
        """
        return normalized_state * self.state_std + self.state_mean

    def get_grid_static_features(self) -> torch.Tensor:
        """
        Get the grid static features tensor.

        Returns
        -------
        torch.Tensor
            Grid static features, shape (num_grid_nodes, static_feature_dim)
        """
        return self.grid_static_features

    def get_state_stats(self) -> Dict[str, torch.Tensor]:
        """
        Get all state statistics as a dictionary.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing 'state_mean', 'state_std', 'diff_mean', 'diff_std'
        """
        return {
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "diff_mean": self.diff_mean,
            "diff_std": self.diff_std,
        }
