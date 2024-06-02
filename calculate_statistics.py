# Standard library
from argparse import ArgumentParser

# Third-party
import xarray as xr

# First-party
from neural_lam import config


def compute_stats(data_array):
    mean = data_array.mean(dim=("time", "grid"))
    std = data_array.std(dim=("time", "grid"))
    return mean, std


def main():
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--data_config",
        type=str,
        default="neural_lam/data_config.yaml",
        help="Path to data config file (default: neural_lam/data_config.yaml)",
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default="normalization.zarr",
        help="Directory where data is stored",
    )
    args = parser.parse_args()

    config_loader = config.Config.from_file(args.data_config)
    state_data = config_loader.process_dataset("state", split="train")
    forcing_data = config_loader.process_dataset("forcing", split="train")

    print("Computing mean and std.-dev. for parameters...", flush=True)
    state_mean, state_std = compute_stats(state_data)

    if forcing_data is not None:
        forcing_mean, forcing_std = compute_stats(forcing_data)
        combined_stats = config_loader["utilities"]["normalization"][
            "combined_stats"
        ]

        if combined_stats is not None:
            for group in combined_stats:
                vars_to_combine = group["vars"]
                means = forcing_mean.sel(variable=vars_to_combine)
                stds = forcing_std.sel(variable=vars_to_combine)

                combined_mean = means.mean(dim="variable")
                combined_std = (stds**2).mean(dim="variable") ** 0.5

                forcing_mean.loc[
                    dict(variable=vars_to_combine)
                ] = combined_mean
                forcing_std.loc[dict(variable=vars_to_combine)] = combined_std

    print(
        "Computing mean and std.-dev. for one-step differences...", flush=True
    )
    state_data_normalized = (state_data - state_mean) / state_std
    state_data_diff_normalized = state_data_normalized.diff(dim="time")
    diff_mean, diff_std = compute_stats(state_data_diff_normalized)

    ds = xr.Dataset(
        {
            "state_mean": state_mean,
            "state_std": state_std,
            "diff_mean": diff_mean,
            "diff_std": diff_std,
        }
    )
    if forcing_data is not None:
        dsf = xr.Dataset(
            {
                "forcing_mean": forcing_mean,
                "forcing_std": forcing_std,
            }
        )
        ds = xr.merge([ds, dsf])

    print("Saving dataset as Zarr...")
    ds = ds.chunk({"variable": -1})
    ds.to_zarr(args.zarr_path, mode="w")
