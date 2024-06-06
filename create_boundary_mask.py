# Standard library
from argparse import ArgumentParser

# Third-party
import numpy as np
import xarray as xr

# First-party
from neural_lam import config


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
        default="data/boundary_mask.zarr",
        help="Path to save the Zarr archive "
        "(default: same directory as data/boundary_mask.zarr)",
    )
    parser.add_argument(
        "--boundaries",
        type=int,
        default=30,
        help="Number of grid-cells to set to True along each boundary",
    )
    args = parser.parse_args()
    data_config = config.Config.from_file(args.data_config)
    mask = np.zeros(list(data_config.grid_shape_state.values.values()))

    # Set the args.boundaries grid-cells closest to each boundary to True
    mask[: args.boundaries, :] = True  # top boundary
    mask[-args.boundaries :, :] = True  # noqa bottom boundary
    mask[:, : args.boundaries] = True  # left boundary
    mask[:, -args.boundaries :] = True  # noqa right boundary

    mask = xr.Dataset({"mask": (["y", "x"], mask)})

    print(f"Saving mask to {args.zarr_path}...")
    mask.to_zarr(args.zarr_path, mode="w")


if __name__ == "__main__":
    main()
