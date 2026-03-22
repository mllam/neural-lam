# Standard library
from datetime import timedelta

# First-party
from neural_lam.utils import get_integer_time


def test_days():
    assert get_integer_time(timedelta(days=14)) == (2, "weeks")


def test_hours():
    assert get_integer_time(timedelta(hours=5)) == (5, "hours")


def test_zero():
    assert get_integer_time(timedelta(0)) == (0, "seconds")


def test_milliseconds():
    assert get_integer_time(timedelta(milliseconds=1000)) == (1, "seconds")


def test_negative():
    assert get_integer_time(timedelta(days=-7)) == (-1, "weeks")


def test_float_days():
    assert get_integer_time(timedelta(days=0.001)) == (86400, "milliseconds")
