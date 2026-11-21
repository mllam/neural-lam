# Third-party
import pytest

# First-party
from neural_lam.metrics import DEFINED_METRICS, get_metric


@pytest.mark.parametrize("metric_name", list(DEFINED_METRICS.keys()))
def test_get_metric_valid(metric_name: str) -> None:
    """`get_metric` retrieves the correct function (case-insensitive)."""
    assert get_metric(metric_name) is DEFINED_METRICS[metric_name]
    assert get_metric(metric_name.upper()) is DEFINED_METRICS[metric_name]


@pytest.mark.parametrize(
    "invalid_name", ["unknown", "rmse", "", "invalid_metric"]
)
def test_get_metric_invalid_raises_value_error(invalid_name: str) -> None:
    """`get_metric` raises ValueError on invalid metric names."""
    with pytest.raises(ValueError, match=f"Unknown metric: '{invalid_name}'"):
        get_metric(invalid_name)
