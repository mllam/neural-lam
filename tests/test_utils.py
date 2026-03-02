from datetime import timedelta

from neural_lam.utils import get_integer_time


def test_weeks():
    assert get_integer_time(timedelta(days=14)) == (2, "weeks")


def test_hours():
    assert get_integer_time(timedelta(hours=5)) == (5, "hours")


def test_milliseconds_promotes_to_seconds():
    assert get_integer_time(timedelta(milliseconds=1000)) == (1, "seconds")


def test_docstring_unknown_example():
    # Matches documented example in utils.py docstring
    assert get_integer_time(timedelta(days=0.001)) == (1, "unknown")