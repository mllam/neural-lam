"""Focused regression tests for backward-compatible metric behavior."""

# Third-party
import matplotlib.pyplot as plt
import torch

# First-party
from neural_lam.metrics import get_metric
from neural_lam.models.ar_model import ARModel


def test_metric_exposes_callable_name():
    """Metric objects should preserve a function-like __name__."""
    for metric_name in (
        "mse",
        "mae",
        "wmse",
        "wmae",
        "nll",
        "crps_gauss",
        "output_std",
    ):
        assert get_metric(metric_name).__name__ == metric_name


def test_aggregate_and_plot_metrics_supports_unknown_metric_keys():
    """Unknown metric keys should still use the generic aggregation path."""

    class MockLogger:
        def log_image(self, key, images):
            pass

    class MockModule:
        def __init__(self):
            self.state_std = torch.tensor([2.0, 3.0])
            self.trainer = type(
                "Trainer",
                (),
                {
                    "is_global_zero": True,
                    "sanity_checking": False,
                    "current_epoch": 0,
                },
            )()
            self.logger = MockLogger()
            self.captured = None

        def all_gather_cat(self, tensor):
            return tensor

        def create_metric_log_dict(self, metric_tensor, prefix, metric_name):
            self.captured = (metric_tensor, prefix, metric_name)
            return {f"{prefix}_{metric_name}": plt.figure()}

    module = MockModule()
    module.aggregate_and_plot_metrics = (
        ARModel.aggregate_and_plot_metrics.__get__(module, MockModule)
    )

    custom_metric_vals = [torch.ones(2, 3, 2)]
    module.aggregate_and_plot_metrics(
        {"custom_metric": custom_metric_vals}, prefix="test"
    )

    metric_tensor, prefix, metric_name = module.captured
    torch.testing.assert_close(
        metric_tensor, torch.tensor([[2.0, 3.0]]).repeat(3, 1)
    )
    assert prefix == "test"
    assert metric_name == "custom_metric"
