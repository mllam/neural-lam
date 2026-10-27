"""Conversion of time deltas to integer-valued time units."""

# Standard library
import datetime


def get_integer_time(tdelta: datetime.timedelta) -> tuple[int, str]:
    """
    Express a :class:`datetime.timedelta` as an integer number of time units.

    Parameters
    ----------
    tdelta : datetime.timedelta
        The time interval to convert. Must be strictly positive.

    Returns
    -------
    int
        Integer value of the timedelta in the largest unit that divides
        it exactly.
    str
        The time unit as a string (``'weeks'``, ``'days'``, ``'hours'``,
        ``'minutes'``, ``'seconds'``, ``'milliseconds'``,
        ``'microseconds'``). Returns ``'unknown'`` if no unit divides
        evenly.

    Raises
    ------
    ValueError
        If ``tdelta`` is not strictly positive.

    Examples
    --------
    >>> from datetime import timedelta
    >>> get_integer_time(timedelta(days=14))
    (2, 'weeks')
    >>> get_integer_time(timedelta(hours=5))
    (5, 'hours')
    >>> get_integer_time(timedelta(milliseconds=1000))
    (1, 'seconds')
    >>> get_integer_time(timedelta(milliseconds=9))
    (9, 'milliseconds')
    >>> get_integer_time(timedelta(microseconds=3))
    (3, 'microseconds')
    """
    if tdelta <= datetime.timedelta(0):
        raise ValueError(f"tdelta must be strictly positive, got {tdelta}")

    total_microseconds = tdelta // datetime.timedelta(microseconds=1)

    units = {
        "weeks": 604_800_000_000,
        "days": 86_400_000_000,
        "hours": 3_600_000_000,
        "minutes": 60_000_000,
        "seconds": 1_000_000,
        "milliseconds": 1_000,
        "microseconds": 1,
    }

    for unit, unit_in_microseconds in units.items():
        if total_microseconds % unit_in_microseconds == 0:
            return total_microseconds // unit_in_microseconds, unit

    return 1, "unknown"
