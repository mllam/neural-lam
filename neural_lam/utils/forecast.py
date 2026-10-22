"""Utilities for constructing forecast datasets."""

# Standard library
from typing import Any

# Third-party
import numpy as np
import torch
import xarray as xr

# Local
from ..datastore.base import BaseDatastore


def create_forecast_dataset(
    predictions: torch.Tensor | np.ndarray,
    analysis_times: np.ndarray | list[Any],
    elapsed_durations: np.ndarray | list[Any],
    datastore: BaseDatastore,
    category: str = "state",
    pred_std: torch.Tensor | np.ndarray | None = None,
    ensemble_members: np.ndarray | list[Any] | None = None,
    split: str = "test",
) -> xr.Dataset:
    """
    Construct an xarray Dataset from weather forecast predictions.

    The dimensions are ``('analysis_time', 'elapsed_forecast_duration',``
    ``['ensemble_member',] 'grid_index', '{category}_feature')``, the order
    ``BaseDatastore.expected_dim_order`` uses for forecast datastores.

    Parameters
    ----------
    predictions : torch.Tensor or np.ndarray
        Model forecast predictions. Expected shape is ``(B, T, N, F)`` for
        deterministic forecasts, or ``(B, T, S, N, F)`` when ensemble members
        are included.
    analysis_times : np.ndarray or list
        Forecast initialization timestamps of length ``B``.
    elapsed_durations : np.ndarray or list
        Elapsed forecast lead times of length ``T``.
    datastore : BaseDatastore
        Datastore providing grid coordinates and feature names.
    category : str, default "state"
        Data category name (usually ``"state"``).
    pred_std : torch.Tensor, np.ndarray, or None, optional
        Standard deviation predictions matching the prediction shape.
    ensemble_members : np.ndarray, list, or None, optional
        Ensemble member identifiers when predictions represent an ensemble.
    split : str, default "test"
        Data split from which to retrieve grid reference coordinates.

    Returns
    -------
    xr.Dataset
        Forecast dataset with standardized coordinates and dimensions.
    """
    if isinstance(predictions, torch.Tensor):
        predictions_np = predictions.detach().cpu().numpy()
    else:
        predictions_np = np.asarray(predictions)

    analysis_times_np = np.atleast_1d(
        np.asarray(analysis_times, dtype="datetime64[ns]")
    )
    elapsed_durations_np = np.atleast_1d(
        np.asarray(elapsed_durations, dtype="timedelta64[ns]")
    )

    ref_da = datastore.get_dataarray(category=category, split=split)
    if ref_da is None:
        raise ValueError(
            f"Unable to retrieve reference DataArray for category '{category}'"
            f" and split '{split}' from datastore."
        )

    var_names = datastore.get_vars_names(category=category)

    has_members = predictions_np.ndim == 5
    if has_members:
        dims = (
            "analysis_time",
            "elapsed_forecast_duration",
            "ensemble_member",
            "grid_index",
            f"{category}_feature",
        )
        if ensemble_members is None:
            ensemble_members = np.arange(predictions_np.shape[2])
        members_np = np.asarray(ensemble_members)
    else:
        dims = (
            "analysis_time",
            "elapsed_forecast_duration",
            "grid_index",
            f"{category}_feature",
        )
        members_np = None

    coords: dict[str, Any] = {
        "analysis_time": analysis_times_np,
        "elapsed_forecast_duration": elapsed_durations_np,
    }
    if members_np is not None:
        coords["ensemble_member"] = members_np

    coords["grid_index"] = ref_da.grid_index
    coords[f"{category}_feature"] = var_names

    # Add 2D valid_time coordinate: analysis_time + elapsed_forecast_duration
    valid_times = (
        analysis_times_np[:, np.newaxis] + elapsed_durations_np[np.newaxis, :]
    )
    coords["valid_time"] = (
        ("analysis_time", "elapsed_forecast_duration"),
        valid_times,
    )

    # Attach spatial coordinates from the reference datastore DataArray
    for coord_name, coord_val in ref_da.coords.items():
        if str(coord_name) not in coords and set(coord_val.dims).issubset(
            {"grid_index"}
        ):
            coords[str(coord_name)] = coord_val

    da = xr.DataArray(
        predictions_np,
        dims=dims,
        coords=coords,
        name=category,
    )
    ds = da.to_dataset()

    if pred_std is not None:
        if isinstance(pred_std, torch.Tensor):
            pred_std_np = pred_std.detach().cpu().numpy()
        else:
            pred_std_np = np.asarray(pred_std)
        ds[f"{category}_std"] = (dims, pred_std_np)

    ds.attrs.update(ref_da.attrs)

    return ds
