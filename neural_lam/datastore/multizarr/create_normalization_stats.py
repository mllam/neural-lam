# Standard library
import argparse
from pathlib import Path

# Third-party
import xarray as xr

# Local
from .store import MultiZarrDatastore

DEFAULT_FILENAME = "normalization.zarr"


def compute_stats(da):
    mean = da.mean(dim=("time", "grid_index"))
    std = da.std(dim=("time", "grid_index"))
    return mean, std


def create_normalization_stats_zarr(
    data_config_path: str,
    zarr_path: str = None,
):
    """
    Compute mean and std.-dev. for state and forcing variables and save them to
    a Zarr file.

    Parameters
    ----------
    data_config_path : str
        Path to data config file.
    zarr_path : str, optional
        Path to save the normalization statistics to. If not provided, the
        statistics are saved to the same directory as the data config file with
        the name `normalization.zarr`.

    """
    if zarr_path is None:
        zarr_path = Path(data_config_path).parent / DEFAULT_FILENAME

    datastore = MultiZarrDatastore(config_path=data_config_path)

    da_state = datastore.get_dataarray(category="state", split="train")
    da_forcing = datastore.get_dataarray(category="forcing", split="train")

    print("Computing mean and std.-dev. for parameters...", flush=True)
    da_state_mean, da_state_std = compute_stats(da_state)

    if da_forcing is not None:
        da_forcing_mean, da_forcing_std = compute_stats(da_forcing)
        combined_stats = datastore._config["utilities"]["normalization"][
            "combined_stats"
        ]

        if combined_stats is not None:
            for group in combined_stats:
                vars_to_combine = group["vars"]

                da_forcing_means = da_forcing_mean.sel(
                    forcing_feature=vars_to_combine
                )
                stds = da_forcing_std.sel(forcing_feature=vars_to_combine)

                combined_mean = da_forcing_means.mean(dim="forcing_feature")
                combined_std = (stds**2).mean(dim="forcing_feature") ** 0.5

                da_forcing_mean.loc[
                    dict(forcing_feature=vars_to_combine)
                ] = combined_mean
                da_forcing_std.loc[
                    dict(forcing_feature=vars_to_combine)
                ] = combined_std
    print(
        "Computing mean and std.-dev. for one-step differences...", flush=True
    )
    state_data_normalized = (da_state - da_state_mean) / da_state_std
    state_data_diff_normalized = state_data_normalized.diff(dim="time")
    diff_mean, diff_std = compute_stats(state_data_diff_normalized)

    ds = xr.Dataset(
        {
            "state_mean": da_state_mean,
            "state_std": da_state_std,
            "state_diff_mean": diff_mean,
            "state_diff_std": diff_std,
        }
    )

    if da_forcing is not None:
        dsf = xr.Dataset(
            {
                "forcing_mean": da_forcing_mean,
                "forcing_std": da_forcing_std,
            }
        )
        ds = xr.merge([ds, dsf])

    ds = ds.chunk({"state_feature": -1, "forcing_feature": -1})
    print("Saving dataset as Zarr...")
    ds.to_zarr(zarr_path, mode="w")


def main():
    parser = argparse.ArgumentParser(
        description="Training arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_config",
        type=str,
        help="Path to data config file",
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default="normalization.zarr",
        help="Directory where data is stored",
    )
    args = parser.parse_args()

    create_normalization_stats_zarr(
        data_config_path=args.data_config, zarr_path=args.zarr_path
    )


if __name__ == "__main__":
    main()
