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
    """
    Pre-compute parameter weights to be used in loss function
    """
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
    parser.add_argument(
        "--combined_forcings",
        action="store_true",
        help="Whether to compute combined stats forcing variables",
    )

    args = parser.parse_args()

    config_loader = config.Config.from_file(args.data_config)

    state_data = config_loader.process_dataset("state", split="train")
    forcing_data = config_loader.process_dataset("forcing", split="train")

    print("Computing mean and std.-dev. for parameters...", flush=True)
    state_mean, state_std = compute_stats(state_data)

    if forcing_data is not None:
        forcing_mean, forcing_std = compute_stats(forcing_data)
        if args.combined_forcings:
            forcing_mean = forcing_mean.mean(dim="variable")
            forcing_std = forcing_std.mean(dim="variable")

    print(
        "Computing mean and std.-dev. for one-step differences...", flush=True
    )
    state_data_diff = state_data.diff(dim="time")
    diff_mean, diff_std = compute_stats(state_data_diff)

    ds = xr.Dataset(
        {
            "state_mean": (["d_features"], state_mean.data),
            "state_std": (["d_features"], state_std.data),
            "diff_mean": (["d_features"], diff_mean.data),
            "diff_std": (["d_features"], diff_std.data),
        }
    )
    if forcing_data is not None:
        dsf = xr.Dataset(
            {
                "forcing_mean": (["d_forcings"], forcing_mean.data),
                "forcing_std": (["d_forcings"], forcing_std.data),
            }
        )
        ds = xr.merge(
            [ds, dsf],
        )
    # Save dataset as Zarr
    print("Saving dataset as Zarr...")
    ds.to_zarr(args.zarr_path, mode="w")


if __name__ == "__main__":
    main()
