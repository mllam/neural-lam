"""Conversion of time deltas to integer-valued time units."""

# Standard library
import datetime


def get_integer_time(tdelta: datetime.timedelta) -> tuple[int, str]:
    """
    Express a :class:`datetime.timedelta` as an integer number of time units.

    Parameters
    ----------
    tdelta : datetime.timedelta
        The time interval to convert.

    Returns
    -------
    int
        Integer value of the timedelta in the largest unit that divides
        it exactly, or ``1`` if no such unit exists.
    str
        The time unit as a string (``'weeks'``, ``'days'``, ``'hours'``,
        ``'minutes'``, ``'seconds'``, ``'milliseconds'``,
        ``'microseconds'``). Returns ``'unknown'`` if no unit divides
        evenly.

    Examples
    --------
    >>> from datetime import timedelta
    >>> get_integer_time(timedelta(days=14))
    (2, 'weeks')
    >>> get_integer_time(timedelta(hours=5))
    (5, 'hours')
    >>> get_integer_time(timedelta(milliseconds=1000))
    (1, 'seconds')
    >>> get_integer_time(timedelta(days=0.001))
    (1, 'unknown')
    """
    total_seconds = tdelta.total_seconds()

    units = {
        "weeks": 604800,
        "days": 86400,
        "hours": 3600,
        "minutes": 60,
        "seconds": 1,
        "milliseconds": 0.001,
        "microseconds": 0.000001,
    }

    for unit, unit_in_seconds in units.items():
        if total_seconds % unit_in_seconds == 0:
            return int(total_seconds / unit_in_seconds), unit

    return 1, "unknown"
