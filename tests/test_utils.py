# Standard library
import datetime

# Third-party
import pytest
import torch
import torch.nn.functional as F

# First-party
from neural_lam.utils import get_integer_time, inverse_softplus


@pytest.mark.parametrize("beta", [1.0, 0.5, 2.0])
def test_inverse_softplus_roundtrip(beta):
    """`inverse_softplus` recovers the input of `softplus`."""
    # Range chosen so softplus(x, beta) stays above the lower clamp
    # log(1+1e-6)/beta for all tested betas (clamp activates around
    # x = -13.8/beta).
    x_orig = torch.linspace(-5, 5, steps=100)
    y = F.softplus(x_orig, beta=beta)
    x_reconstructed = inverse_softplus(y, beta=beta)
    torch.testing.assert_close(x_orig, x_reconstructed)


def test_inverse_softplus_near_zero_is_finite():
    """Near-zero inputs are clamped so the log path cannot produce NaN/-Inf."""
    y_near_zero = torch.tensor([1e-7, 1e-6])
    x_near_zero = inverse_softplus(y_near_zero)
    assert torch.isfinite(x_near_zero).all()


@pytest.mark.parametrize("threshold", [20.0, 5.0])
def test_inverse_softplus_above_threshold_is_identity(threshold):
    """Values above `threshold` bypass the log path and return unchanged."""
    y_high = torch.tensor([threshold + 5.0, threshold + 30.0])
    x_high = inverse_softplus(y_high, threshold=threshold)
    torch.testing.assert_close(y_high, x_high)


@pytest.mark.parametrize(
    "tdelta, expected",
    [
        (datetime.timedelta(weeks=2), (2, "weeks")),
        (datetime.timedelta(days=14), (2, "weeks")),
        (datetime.timedelta(days=3), (3, "days")),
        (datetime.timedelta(hours=24), (1, "days")),
        (datetime.timedelta(hours=5), (5, "hours")),
        (datetime.timedelta(minutes=60), (1, "hours")),
        (datetime.timedelta(minutes=15), (15, "minutes")),
        (datetime.timedelta(seconds=60), (1, "minutes")),
        (datetime.timedelta(seconds=45), (45, "seconds")),
        (datetime.timedelta(milliseconds=1000), (1, "seconds")),
        (datetime.timedelta(microseconds=1000), (1, "milliseconds")),
    ],
)
def test_get_integer_time_standard_units(tdelta, expected):
    """get_integer_time returns the largest evenly dividing unit."""
    assert get_integer_time(tdelta) == expected


@pytest.mark.parametrize("ms", range(1, 11))
def test_get_integer_time_milliseconds_precision(ms):
    """Milliseconds 1..10 should not suffer from float modulo errors."""
    td = datetime.timedelta(milliseconds=ms)
    assert get_integer_time(td) == (ms, "milliseconds")


@pytest.mark.parametrize("us", range(1, 11))
def test_get_integer_time_microseconds_precision(us):
    """Microseconds 1..10 should not suffer from float modulo errors."""
    td = datetime.timedelta(microseconds=us)
    assert get_integer_time(td) == (us, "microseconds")


@pytest.mark.parametrize(
    "invalid_tdelta",
    [
        datetime.timedelta(0),
        datetime.timedelta(seconds=-1),
        datetime.timedelta(hours=-5),
    ],
)
def test_get_integer_time_non_positive_raises_value_error(invalid_tdelta):
    """Non-positive timedeltas should raise ValueError."""
    with pytest.raises(ValueError, match="tdelta must be strictly positive"):
        get_integer_time(invalid_tdelta)
