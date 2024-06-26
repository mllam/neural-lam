# Standard library
from argparse import ArgumentParser

# Third-party
import xarray as xr

# First-party
from neural_lam.datastore.multizarr import MultiZarrDatastore


DEFAULT_PATH = "tests/datastore_configs/multizarr.danra.yaml"


def compute_stats(da):
    mean = da.mean(dim=("time", "grid_index"))
    std = da.std(dim=("time", "grid_index"))
    return mean, std


def main():
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--data_config",
        type=str,
        default=DEFAULT_PATH,
        help=f"Path to data config file (default: {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default="data/normalization.zarr",
        help="Directory where data is stored",
    )
    args = parser.parse_args()
    
    datastore = MultiZarrDatastore(config_path=args.data_config)

    da_state = datastore.get_dataarray(category="state", split="train")
    da_forcing = datastore.get_dataarray(category="forcing", split="train")

    print("Computing mean and std.-dev. for parameters...", flush=True)
    da_state_mean, da_state_std = compute_stats(da_state)

    if da_forcing is not None:
        da_forcing_mean, da_forcing_std = compute_stats(da_forcing)
        combined_stats = datastore._config["utilities"]["normalization"]["combined_stats"]

        if combined_stats is not None:
            for group in combined_stats:
                vars_to_combine = group["vars"]
                import ipdb; ipdb.set_trace()
                means = da_forcing_mean.sel(variable=vars_to_combine)
                stds = da_forcing_std.sel(variable=vars_to_combine)

                combined_mean = means.mean(dim="variable")
                combined_std = (stds**2).mean(dim="variable") ** 0.5

                da_forcing_mean.loc[dict(variable=vars_to_combine)] = combined_mean
                da_forcing_std.loc[dict(variable=vars_to_combine)] = combined_std

        window = datastore._config["forcing"]["window"]

        da_forcing_mean = xr.concat([da_forcing_mean] * window, dim="window").stack(
            forcing_variable=("variable", "window")
        )
        da_forcing_std = xr.concat([da_forcing_std] * window, dim="window").stack(
            forcing_variable=("variable", "window")
        )
        vars = da_forcing["variable"].values.tolist()
        window = datastore._config["forcing"]["window"]
        forcing_vars = [f"{var}_{i}" for var in vars for i in range(window)]

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
            "diff_mean": diff_mean,
            "diff_std": diff_std,
        }
    )
    if da_forcing is not None:
        dsf = (
            xr.Dataset(
                {
                    "forcing_mean": da_forcing_mean,
                    "forcing_std": da_forcing_std,
                }
            )
            .reset_index(["forcing_variable"])
            .drop_vars(["variable", "window"])
            .assign_coords(forcing_variable=forcing_vars)
        )
        ds = xr.merge([ds, dsf])

    ds = ds.chunk({"variable": -1, "forcing_variable": -1})
    print("Saving dataset as Zarr...")
    ds.to_zarr(args.zarr_path, mode="w")


if __name__ == "__main__":
    main()
