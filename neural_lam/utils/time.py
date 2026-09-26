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
    times : np.ndarray
        1D array of datetime64 or timedelta64 values.

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


def _requested_time_bounds(
    da_requested: xr.DataArray,
    da_available: xr.DataArray,
    da_requested_is_forecast: bool,
    da_available_is_forecast: bool,
    num_past_steps: int,
    num_future_steps: int,
    requested_max_lead: np.timedelta64 | None,
) -> tuple[np.datetime64, np.datetime64]:
    """Return the first and last ``da_requested`` time ``da_available`` covers.

    The window is measured along whichever ``da_available`` axis
    :meth:`WeatherDataset._window_forcing_in_time` walks: ``time`` for
    analysis data, ``elapsed_forecast_duration`` for a forecast.

    Parameters
    ----------
    da_requested, da_available : xr.DataArray
        See :func:`get_time_crop_slice`.
    da_requested_is_forecast, da_available_is_forecast : bool
        See :func:`get_time_crop_slice`.
    num_past_steps, num_future_steps : int
        See :func:`get_time_crop_slice`.
    requested_max_lead : np.timedelta64 or None
        See :func:`get_time_crop_slice`.

    Returns
    -------
    tuple of np.datetime64
        Inclusive ``[first, last]`` bounds on the requested times.
    """
    if not da_requested_is_forecast:
        requested_max_lead = np.timedelta64(0, "ns")
    elif requested_max_lead is None:
        requested_max_lead = da_requested.elapsed_forecast_duration.values.max()

    if da_available_is_forecast:
        times_available = da_available.analysis_time.values
        leads_available = da_available.elapsed_forecast_duration.values
        # Lead spacing sizes the window, not analysis spacing: stepping back
        # one launch buys `analysis / lead` window steps.
        step_window = get_time_step(leads_available)
        # The init time is never earlier than the requested time itself, so
        # the first launch alone bounds the start.
        first = (
            times_available.min()
            + leads_available.min()
            + num_past_steps * step_window
        )
        last = (
            times_available.max()
            + leads_available.max()
            - requested_max_lead
            - num_future_steps * step_window
        )
    else:
        times_available = da_available.time.values
        step_window = get_time_step(times_available)
        first = times_available.min() + num_past_steps * step_window
        last = (
            times_available.max()
            - requested_max_lead
            - num_future_steps * step_window
        )

    return first, last


def get_time_crop_slice(
    da_requested: xr.DataArray,
    da_available: xr.DataArray,
    da_requested_is_forecast: bool = False,
    da_available_is_forecast: bool = False,
    num_past_steps: int = 1,
    num_future_steps: int = 1,
    requested_max_lead: np.timedelta64 | None = None,
) -> tuple[str, slice]:
    """Return the ``da_requested`` dimension and slice ``da_available`` covers.

    Callers holding several dataarrays on the same time axis (interior state
    and forcing) apply this one slice to all of them so they stay aligned.

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
        Window size around each requested time, measured in ``da_available``
        steps.
    requested_max_lead : np.timedelta64, optional
        Largest lead read from a forecast ``da_requested`` per sample; each
        ``analysis_time`` needs coverage out to it. Defaults to the full
        forecast length.

    Returns
    -------
    tuple of (str, slice)
        The dimension to crop along (``"analysis_time"`` for a forecast
        ``da_requested``, else ``"time"``) and the positional slice to keep.

    Raises
    ------
    ValueError
        If no requested time is covered at all.
    """
    crop_dim = "analysis_time" if da_requested_is_forecast else "time"
    requested_tvals = da_requested[crop_dim].values
    first, last = _requested_time_bounds(
        da_requested,
        da_available,
        da_requested_is_forecast,
        da_available_is_forecast,
        num_past_steps,
        num_future_steps,
        requested_max_lead,
    )

    first_valid_idx = int(np.searchsorted(requested_tvals, first, side="left"))
    last_valid_idx_plus_one = int(
        np.searchsorted(requested_tvals, last, side="right")
    )
    if first_valid_idx >= last_valid_idx_plus_one:
        raise ValueError(
            f"`da_available` covers no `da_requested` `{crop_dim}` in "
            f"[{first}, {last}]; cannot align."
        )
    return crop_dim, slice(first_valid_idx, last_valid_idx_plus_one)


def apply_time_crop(
    da: xr.DataArray, crop_dim: str, crop_slice: slice
) -> xr.DataArray:
    """Apply a :func:`get_time_crop_slice` result to ``da``, logging removals.

    Parameters
    ----------
    da : xr.DataArray
        Dataarray to crop.
    crop_dim : str
        Dimension to crop along.
    crop_slice : slice
        Positional slice to keep.

    Returns
    -------
    xr.DataArray
        ``da`` cropped, or unchanged when the slice keeps everything.
    """
    n_removed_begin = crop_slice.start
    n_removed_end = da.sizes[crop_dim] - crop_slice.stop
    if n_removed_begin == 0 and n_removed_end == 0:
        return da

    log_on_rank_zero(
        f"Cropping `{da.name or 'dataarray'}` to align with the available "
        f"time coverage: removed {n_removed_begin} {crop_dim} steps at start "
        f"and {n_removed_end} at the end.",
        level="warning",
    )
    return da.isel({crop_dim: crop_slice})
