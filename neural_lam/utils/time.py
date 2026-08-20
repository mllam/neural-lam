"""Time-delta conversion and interior/boundary time-axis alignment."""

# Standard library
import datetime

# Third-party
import numpy as np
import xarray as xr

# Local
from .logging import log_on_rank_zero


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


def get_time_step(times: np.ndarray) -> np.timedelta64:
    """Calculate the (constant) time step from a 1D time array.

    Parameters
    ----------
    times : array-like
        A 1D array of datetime64 (or timedelta64) values to compute the
        step from.

    Returns
    -------
    time_step : np.timedelta64
        The constant spacing between successive values.

    Raises
    ------
    ValueError
        If fewer than two values are given, or if the spacing is not
        constant.
    """
    times = np.asarray(times)
    if times.size < 2:
        raise ValueError(
            "Cannot determine a time step from a time axis with "
            f"{times.size} value(s); at least 2 are required."
        )
    time_diffs = np.diff(times)
    if not np.all(time_diffs == time_diffs[0]):
        raise ValueError(
            "Inconsistent time steps in data. "
            f"Found different time steps: {np.unique(time_diffs)}"
        )
    return time_diffs[0]


def check_time_overlap(
    da_requested: xr.DataArray,
    da_available: xr.DataArray,
    da_requested_is_forecast: bool = False,
    da_available_is_forecast: bool = False,
    num_past_steps: int = 1,
    num_future_steps: int = 1,
) -> None:
    """Check that the time coverage of ``da_available`` is wide enough to
    support a windowed lookup driven by ``da_requested`` times with the
    given past/future window sizes.

    Parameters
    ----------
    da_requested : xr.DataArray
        Driving dataarray whose times must be supported (typically interior
        state).
    da_available : xr.DataArray
        Dataarray that must cover the requested windows (typically boundary
        forcing).
    da_requested_is_forecast, da_available_is_forecast : bool
        Whether each side is in forecast mode (``analysis_time`` +
        ``elapsed_forecast_duration`` dims) instead of plain ``time``.
    num_past_steps, num_future_steps : int
        Window size around each ``da_requested`` time, measured in
        ``da_available`` steps.

    Raises
    ------
    ValueError
        If ``da_available`` does not cover the required time range.
    """
    if da_requested_is_forecast:
        times_requested = da_requested.analysis_time
    else:
        times_requested = da_requested.time
    time_min_requested = times_requested.min().values
    time_max_requested = times_requested.max().values

    if da_available_is_forecast:
        times_available = da_available.analysis_time
        time_min_available = times_available.min().values
        time_max_available = times_available.max().values

        time_step_available = get_time_step(times_available.values)
        time_step_requested = get_time_step(times_requested.values)
        max_lead = da_available.elapsed_forecast_duration.values.max()

        analysis_offset = max(
            time_step_requested, num_past_steps * time_step_available
        )
        required_time_min = time_min_requested - analysis_offset
        required_time_max = time_max_requested - (
            max_lead - num_future_steps * time_step_available
        )
    else:
        times_available = da_available.time
        time_min_available = times_available.min().values
        time_max_available = times_available.max().values
        time_step_available = get_time_step(times_available.values)

        required_time_min = (
            time_min_requested - num_past_steps * time_step_available
        )
        required_time_max = (
            time_max_requested + num_future_steps * time_step_available
        )

    if time_min_available > required_time_min:
        raise ValueError(
            "`da_available` starts too late to cover the requested window. "
            f"Required start: {required_time_min}, "
            f"but `da_available` starts at {time_min_available}."
        )

    if time_max_available < required_time_max:
        raise ValueError(
            "`da_available` ends too early to cover the requested window. "
            f"Required end: {required_time_max}, "
            f"but `da_available` ends at {time_max_available}."
        )


def crop_time_if_needed(
    da_requested: xr.DataArray,
    da_available: xr.DataArray,
    da_requested_is_forecast: bool = False,
    da_available_is_forecast: bool = False,
    num_past_steps: int = 1,
    num_future_steps: int = 1,
) -> xr.DataArray:
    """Trim the leading/trailing times from ``da_requested`` so that
    ``da_available`` covers every needed window. If ``check_time_overlap``
    already passes, ``da_requested`` is returned unchanged. A forecast-mode
    ``da_requested`` is cropped along ``analysis_time`` (dropping whole
    launches), an analysis-mode one along ``time``; either way the removal
    is logged.

    Parameters mirror :func:`check_time_overlap`.

    Returns
    -------
    xr.DataArray
        Possibly cropped ``da_requested``.
    """
    if da_requested is None or da_available is None:
        return da_requested

    try:
        check_time_overlap(
            da_requested,
            da_available,
            da_requested_is_forecast,
            da_available_is_forecast,
            num_past_steps,
            num_future_steps,
        )
        return da_requested
    except ValueError:
        crop_dim = "analysis_time" if da_requested_is_forecast else "time"
        requested_tvals = da_requested[crop_dim].values
        if da_available_is_forecast:
            available_tvals = da_available.analysis_time.values
        else:
            available_tvals = da_available.time.values

        available_dt = get_time_step(available_tvals)
        if da_available_is_forecast:
            requested_dt = get_time_step(requested_tvals)
            max_lead = da_available.elapsed_forecast_duration.values.max()
            analysis_offset = max(requested_dt, num_past_steps * available_dt)
            required_min = available_tvals[0] + analysis_offset
            required_max = (
                available_tvals[-1] + max_lead - num_future_steps * available_dt
            )
        else:
            required_min = available_tvals[0] + num_past_steps * available_dt
            required_max = available_tvals[-1] - num_future_steps * available_dt

        first_valid_idx = int(
            np.searchsorted(requested_tvals, required_min, side="left")
        )
        last_valid_idx_plus_one = int(
            np.searchsorted(requested_tvals, required_max, side="right")
        )
        if first_valid_idx >= last_valid_idx_plus_one:
            raise ValueError(
                "`da_available` covers no `da_requested` time in "
                f"[{required_min}, {required_max}]; cannot align."
            )
        n_removed_begin = first_valid_idx
        n_removed_end = len(requested_tvals) - last_valid_idx_plus_one

        if n_removed_begin > 0 or n_removed_end > 0:
            log_on_rank_zero(
                f"Cropping `da_requested` to align with `da_available`: "
                f"removed {n_removed_begin} {crop_dim} steps at start and "
                f"{n_removed_end} at the end.",
                level="warning",
            )
            da_requested = da_requested.isel(
                {crop_dim: slice(first_valid_idx, last_valid_idx_plus_one)}
            )
        return da_requested
