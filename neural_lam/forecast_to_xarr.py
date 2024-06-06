# Standard library
import os

# Third-party
import numpy as np
import torch
import xarray as xa
from tqdm import tqdm

# First-party
from neural_lam import constants
from neural_lam.models.graph_efm import GraphEFM

FC_DIR_PATH = "saved_forecasts"


def get_var_dims(save_ensemble):
    """
    Get dimension names for atmospheric and surface variables
    """
    atm_dims = (
        "time",
        "prediction_timedelta",
        "longitude",
        "latitude",
        "level",
    )
    sur_dims = (
        "time",
        "prediction_timedelta",
        "longitude",
        "latitude",
    )

    if save_ensemble:
        atm_dims = ("realization",) + atm_dims
        sur_dims = ("realization",) + sur_dims

    return atm_dims, sur_dims


def forecast_to_xds(
    forecast_tensor,
    batch_init_times,
    coords,
    var_filter_list,
    level_filter_list,
    time_enc_unit,
):
    """
    Turn a pytorch tensor representing a forecast into a saveable xarray.Dataset

    forecast_tensor: (B, (S), pred_steps, num_grid_nodes, d_f)
    """
    # Figure out if ensemble (S-dimension exists)
    save_ensemble = len(forecast_tensor.shape) == 5

    full_fc = forecast_tensor.cpu().numpy()
    # (B, (S), pred_steps, num_grid_nodes, d_f)
    # Now on CPU, numpy

    # Reshape to grid shape
    # Note: this reshape works with or without S-dimension
    full_fc_grid = full_fc.reshape(
        *full_fc.shape[:-2],
        *constants.GRID_SHAPE,
        full_fc.shape[-1],
    )  # (B, (S), pred_steps, num_lon, num_lat, d_f)

    if save_ensemble:
        # Transpose first dimensions, so they are (realization, time, ...)
        full_fc_grid = np.moveaxis(full_fc_grid, 1, 0)

    fc_sur = full_fc_grid[..., -len(constants.SURFACE_PARAMS) :]
    # (..., num_sur_vars)
    fc_atm = full_fc_grid[..., : -len(constants.SURFACE_PARAMS)]
    # (..., num_atm_vars * num_levels)

    fc_sur_list = np.split(fc_sur, len(constants.SURFACE_PARAMS), axis=-1)
    fc_sur_list = [fc.squeeze(-1) for fc in fc_sur_list]
    # list of ((S), B, pred_steps, num_lon, num_lat)

    fc_atm_list = np.split(fc_atm, len(constants.ATMOSPHERIC_PARAMS), axis=-1)
    # list of ((S), B, pred_steps, num_lon, num_lat, num_levels)

    # Turn whole forecast into xr.Dataset
    atm_dims, sur_dims = get_var_dims(save_ensemble)
    fc_var_dict = dict(
        zip(
            constants.ATMOSPHERIC_PARAMS,
            ((atm_dims, var_vals) for var_vals in fc_atm_list),
        )
    ) | dict(
        zip(
            constants.SURFACE_PARAMS,
            ((sur_dims, var_vals) for var_vals in fc_sur_list),
        )
    )

    # Create dataset
    batch_xds = xa.Dataset(
        fc_var_dict,
        coords={
            "time": batch_init_times,
        }
        | {c: v.values for c, v in coords if c != "time"},
    )
    batch_xds.time.encoding["units"] = time_enc_unit

    # Filter batch dataset
    filtered_batch_xds = filter_xdataset(
        batch_xds, var_filter_list, level_filter_list
    )

    # Optionally compute and add wind speeds
    if (
        "u_component_of_wind" in filtered_batch_xds
        and "v_component_of_wind" in filtered_batch_xds
    ):
        wind_speed = np.sqrt(
            filtered_batch_xds["u_component_of_wind"] ** 2
            + filtered_batch_xds["v_component_of_wind"] ** 2
        )
        filtered_batch_xds["wind_speed"] = wind_speed
    if (
        "10m_u_component_of_wind" in filtered_batch_xds
        and "10m_v_component_of_wind" in filtered_batch_xds
    ):
        wind_speed = np.sqrt(
            filtered_batch_xds["10m_u_component_of_wind"] ** 2
            + filtered_batch_xds["10m_v_component_of_wind"] ** 2
        )
        filtered_batch_xds["10m_wind_speed"] = wind_speed

    return filtered_batch_xds


def parse_filters(var_filter_str, level_filter_str):
    """
    Parse and check correctness of variable and level filters given as strings.
    """
    # Variable filter
    if var_filter_str is None:
        var_list = None
    else:
        # String to list
        var_list_short = [
            var_str.strip() for var_str in var_filter_str.split(",")
        ]

        # Check that all variables are forecasted
        for var_str in var_list_short:
            assert (
                var_str in constants.ATMOSPHERIC_PARAMS_SHORT
                or var_str in constants.SURFACE_PARAMS_SHORT
            ), f"Can not save unknown variable: {var_str}"

        param_name_lookup = dict(
            zip(constants.SURFACE_PARAMS_SHORT, constants.SURFACE_PARAMS)
        ) | dict(
            zip(
                constants.ATMOSPHERIC_PARAMS_SHORT, constants.ATMOSPHERIC_PARAMS
            )
        )
        var_list = [
            param_name_lookup[short_name] for short_name in var_list_short
        ]

    # Level filter
    if level_filter_str is None:
        level_list = None
    else:
        level_list = [
            int(level_str.strip()) for level_str in level_filter_str.split(",")
        ]
        for level in level_list:
            assert (
                level in constants.PRESSURE_LEVELS
            ), f"Can not save unknown pressure level: {level}"

    return var_list, level_list


def filter_xdataset(xds, var_filter_list, level_filter_list):
    """
    Filter out selected variables and levels from xarray.Dataset
    """
    if var_filter_list is not None:
        xds = xds[var_filter_list]

    if level_filter_list is not None:
        # Need nearest method to keep surface variables
        xds = xds.sel(level=level_filter_list, method="nearest")

    return xds


@torch.no_grad()
def forecast_to_xarr(
    model,
    dataloader,
    name,
    device_name,
    var_filter=None,
    level_filter=None,
    ens_size=5,
):
    """
    Produce forecasts for each sample in the data_loader, using model

    model: model to produce forecasts with
    dataloader: non-shuffling dataloader for evaluation set
    name: name to save zarr as (without .zarr)
    device_name: name of device to use for forecasting
    var_filter: string, comma-separated list of variables to save,
        or None to save all
    """
    # Parse var_filter
    var_filter_list, level_filter_list = parse_filters(var_filter, level_filter)

    # Set up device, need to handle manually here
    device = torch.device(device_name)
    model = model.to(device)

    # Set up save path
    os.makedirs(FC_DIR_PATH, exist_ok=True)
    fc_path = os.path.join(FC_DIR_PATH, f"{name}.zarr")

    # Get coordinates from array used in dataset
    dataset = dataloader.dataset
    data_mean = dataset.data_mean.to(device)
    data_std = dataset.data_std.to(device)
    ds_xda = dataset.atm_xda

    # Set up xarray with zarr backend
    pred_hours = 6 * (np.arange(dataset.pred_length) + 1)
    pred_timedeltas = [
        np.timedelta64(dh, "h").astype("timedelta64[ns]") for dh in pred_hours
    ]

    # Figure out if we should do ensemble forecasting
    save_ensemble = isinstance(model, GraphEFM)

    # Set up dimensions for dataset
    atm_dims, sur_dims = get_var_dims(save_ensemble)
    atm_empty_shape = (
        0,
        dataset.pred_length,
        *constants.GRID_SHAPE,
        len(constants.PRESSURE_LEVELS),
    )
    sur_empty_shape = (
        0,
        dataset.pred_length,
        *constants.GRID_SHAPE,
    )
    ds_coords = {
        "time": np.array([], dtype="datetime64[ns]"),
        "prediction_timedelta": pred_timedeltas,
        "longitude": ds_xda.coords["longitude"].values,
        "latitude": ds_xda.coords["latitude"].values,
        "level": ds_xda.coords["level"].values,
    }

    if save_ensemble:
        # Add on realization (ens. member) dim.
        atm_empty_shape = (ens_size,) + atm_empty_shape
        sur_empty_shape = (ens_size,) + sur_empty_shape
        ds_coords["realization"] = np.arange(ens_size)

    forecast_xds = xa.Dataset(
        {
            var_name: (
                atm_dims,
                np.zeros(atm_empty_shape),
            )
            for var_name in constants.ATMOSPHERIC_PARAMS
        }
        | {  # Dict union
            var_name: (
                sur_dims,
                np.zeros(sur_empty_shape),
            )
            for var_name in constants.SURFACE_PARAMS
        },
        coords=ds_coords,
    )
    # Need to set this encoding to save/load correct times from disk
    time_enc_unit = "nanoseconds since 1970-01-01"
    forecast_xds.time.encoding["units"] = time_enc_unit

    # Filter to selected
    filtered_xds = filter_xdataset(
        forecast_xds, var_filter_list, level_filter_list
    )

    # Set up wind variables
    if (
        "u_component_of_wind" in filtered_xds
        and "v_component_of_wind" in filtered_xds
    ):
        # Use same empty shape
        filtered_xds["wind_speed"] = filtered_xds["u_component_of_wind"]
    if (
        "10m_u_component_of_wind" in filtered_xds
        and "10m_v_component_of_wind" in filtered_xds
    ):
        # Use same empty shape
        filtered_xds["10m_wind_speed"] = filtered_xds["10m_u_component_of_wind"]

    # Set up chunking
    atm_chunking = (1, -1, -1, -1, -1)
    sur_chunking = (1, -1, -1, -1)
    if save_ensemble:
        # All members in same chunk
        atm_chunking = (-1,) + atm_chunking
        sur_chunking = (-1,) + sur_chunking

    chunk_encoding = dict(  # pylint: disable=consider-using-dict-comprehension
        [
            (
                (v, {"chunks": atm_chunking})
                if v in constants.ATMOSPHERIC_PARAMS + ["wind_speed"]
                else (v, {"chunks": sur_chunking})
            )
            for v in filtered_xds
        ]
    )

    # Overwrite if exists
    filtered_xds.to_zarr(fc_path, mode="w", encoding=chunk_encoding)

    # Compute all init times
    start_init_time = ds_xda.coords["time"].values[1]
    end_init_time = start_init_time + np.timedelta64(len(dataset) * 12, "h")
    init_times = np.arange(
        start_init_time, end_init_time, np.timedelta64(12, "h")
    ).astype("datetime64[ns]")

    # Iterate over dataset and produce forecasts
    for batch in tqdm(dataloader):
        # Send to device
        batch = tuple(t.to(device) for t in batch)

        # Forecast
        if save_ensemble:
            init_states, target_states, forcing_features = batch

            batch_forecast, _ = model.sample_trajectories(
                init_states,
                forcing_features,
                target_states,
                ens_size,
            )
            # (B, S, pred_steps, num_grid_nodes, d_f)
        else:
            batch_forecast, _, _ = model.common_step(batch)

        # Rescale to original data scaling
        batch_forecast_rescaled = batch_forecast * data_std + data_mean
        # (B, (S), pred_steps, num_grid_nodes, d_f)

        # Get init times for batch
        batch_size = batch_forecast.shape[0]
        batch_init_times = init_times[:batch_size]
        init_times = init_times[batch_size:]  # Drop used times

        batch_xds = forecast_to_xds(
            batch_forecast_rescaled,
            batch_init_times,
            forecast_xds.coords.items(),
            var_filter_list,
            level_filter_list,
            time_enc_unit,
        )

        # Save to existing zarr using append_dim="time"
        batch_xds.to_zarr(fc_path, append_dim="time")
