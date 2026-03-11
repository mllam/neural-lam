from datetime import timedelta

import pytest

from neural_lam.utils import get_integer_time


@pytest.mark.parametrize(
    "tdelta, expected",
    [
        (timedelta(days=14), (2, "weeks")),
        (timedelta(hours=5), (5, "hours")),
        (timedelta(milliseconds=1000), (1, "seconds")),
    ],
)
def test_get_integer_time_doc_examples(tdelta, expected):
    """The function should match the documented examples in its docstring."""
    assert get_integer_time(tdelta) == expected


def test_get_integer_time_zero_timedelta():
    """Zero timedelta is exactly divisible by any unit; the implementation
    returns weeks with value 0 as the first matching unit."""
    assert get_integer_time(timedelta(0)) == (0, "weeks")


@pytest.mark.parametrize(
    "tdelta",
    [
        timedelta(days=0.001),  # From the docstring example
        timedelta(seconds=1, milliseconds=500),  # 1.5 seconds
    ],
)
def test_get_integer_time_no_exact_integer_unit(tdelta):
    """When no unit can represent the timedelta as an integer, 'unknown'
    should be returned as documented."""
    value, unit = get_integer_time(tdelta)
    assert value == 1
    assert unit == "unknown"


@pytest.mark.parametrize(
    "tdelta, expected",
    [
        (timedelta(microseconds=1), (1, "microseconds")),
        (timedelta(milliseconds=250_000), (250, "seconds")),
        (timedelta(hours=-3), (-3, "hours")),
    ],
)
def test_get_integer_time_edge_cases(tdelta, expected):
    """Additional edge cases for robustness, including very small and negative
    timedeltas."""
    assert get_integer_time(tdelta) == expected

