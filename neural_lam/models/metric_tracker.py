# Standard library
from typing import Any, Dict, List

# Third-party
import torch


class MetricTracker:
    """
    Tracks validation and test metrics during model evaluation.

    This class manages the collection and storage of metrics across
    multiple batches during validation and testing phases.
    """

    def __init__(self, output_std: bool = False):
        """
        Initialize the MetricTracker.

        Parameters
        ----------
        output_std : bool, optional
            Whether the model outputs standard deviation, by default False
        """
        self.output_std = output_std
        self.val_metrics: Dict[str, List] = {
            "mse": [],
        }
        self.test_metrics: Dict[str, List] = {
            "mse": [],
            "mae": [],
        }
        if self.output_std:
            self.test_metrics["output_std"] = []  # Treat as metric

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps: List[Any] = []

    def add_val_metric(self, metric_name: str, metric_value: torch.Tensor):
        """
        Add a validation metric value.

        Parameters
        ----------
        metric_name : str
            Name of the metric (e.g., 'mse', 'mae')
        metric_value : torch.Tensor
            The metric value tensor to store
        """
        if metric_name not in self.val_metrics:
            self.val_metrics[metric_name] = []
        self.val_metrics[metric_name].append(metric_value)

    def add_test_metric(self, metric_name: str, metric_value: torch.Tensor):
        """
        Add a test metric value.

        Parameters
        ----------
        metric_name : str
            Name of the metric (e.g., 'mse', 'mae', 'output_std')
        metric_value : torch.Tensor
            The metric value tensor to store
        """
        if metric_name not in self.test_metrics:
            self.test_metrics[metric_name] = []
        self.test_metrics[metric_name].append(metric_value)

    def add_spatial_loss_map(self, spatial_loss: torch.Tensor):
        """
        Add a spatial loss map.

        Parameters
        ----------
        spatial_loss : torch.Tensor
            Spatial loss tensor, typically (B, N_log, num_grid_nodes)
        """
        self.spatial_loss_maps.append(spatial_loss)

    def clear_val_metrics(self):
        """Clear all validation metrics."""
        for metric_list in self.val_metrics.values():
            metric_list.clear()

    def clear_test_metrics(self):
        """Clear all test metrics."""
        for metric_list in self.test_metrics.values():
            metric_list.clear()

    def clear_spatial_loss_maps(self):
        """Clear all spatial loss maps."""
        self.spatial_loss_maps.clear()

    def get_val_metrics(self) -> Dict[str, List]:
        """
        Get all validation metrics.

        Returns
        -------
        Dict[str, List]
            Dictionary mapping metric names to lists of tensors
        """
        return self.val_metrics

    def get_test_metrics(self) -> Dict[str, List]:
        """
        Get all test metrics.

        Returns
        -------
        Dict[str, List]
            Dictionary mapping metric names to lists of tensors
        """
        return self.test_metrics

    def get_spatial_loss_maps(self) -> List[Any]:
        """
        Get all spatial loss maps.

        Returns
        -------
        List[Any]
            List of spatial loss map tensors
        """
        return self.spatial_loss_maps
