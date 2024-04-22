# Standard library
import os
from argparse import ArgumentParser

# Third-party
import numpy as np
import xarray as xr

# First-party
from neural_lam import constants


def main():
    """Create the static features for the neural network."""
    parser = ArgumentParser(description="Static features arguments")
    parser.add_argument(
        "--xdim",
        type=str,
        default="x",
        help="Name of the x-dimension in the dataset (default: x)",
    )
    parser.add_argument(
        "--ydim",
        type=str,
        default="y",
        help="Name of the x-dimension in the dataset (default: y)",
    )
    parser.add_argument(
        "--zdim",
        type=str,
        default="z",
        help="Name of the x-dimension in the dataset (default: z)",
    )
    parser.add_argument(
        "--field_names",
        nargs="+",
        default=["HSURF", "FI", "HFL"],
        help=(
            "Names of the fields to extract from the .nc file "
            '(default: ["HSURF", "FI", "HFL"])'
        ),
    )
    parser.add_argument(
        "--boundaries",
        type=int,
        default=30,
        help=(
            "Number of grid-cells closest to each boundary to mask "
            "(default: 30)"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cosmo",
        help=("Name of the dataset (default: cosmo)"),
    )

    args = parser.parse_args()

    ds = xr.open_zarr(constants.EXAMPLE_FILE).isel(time=0)

    np_fields = []

    for var_name in args.field_names:
        # scale the variable to [0, 1]
        ds[var_name] = (ds[var_name] - ds[var_name].min()) / (
            ds[var_name].max() - ds[var_name].min()
        )

        if args.zdim not in ds[var_name].dims:
            field_2d = ds[var_name].transpose(args.xdim, args.ydim).values
            # add a dummy dimension
            np_fields.append(np.expand_dims(field_2d, axis=-1))
        else:
            np_fields.append(
                ds[var_name]
                .sel({args.zdim: constants.VERTICAL_LEVELS})
                .transpose(args.xdim, args.ydim, args.zdim)
                .values
            )
    np_fields = np.concatenate(np_fields, axis=-1)  # (N_x, N_y, N_fields)

    outdir = os.path.join("data", args.dataset, "static/")

    # Save the numpy array to a .npy file
    np.save(outdir + "reference_geopotential_pressure.npy", np_fields)

    # Get the dimensions of the dataset
    dims = ds.sizes
    x_dim, y_dim = ds.sizes[args.xdim], ds.sizes[args.ydim]

    # Create a 2D meshgrid for x and y indices
    x_grid, y_grid = np.indices((x_dim, y_dim))

    # Stack the 2D arrays into a 3D array with x and y as the first dimension
    grid_xy = np.stack((y_grid, x_grid))

    np.save(outdir + "nwp_xy.npy", grid_xy)  # (2, N_x, N_y)

    # Create a mask with the same dimensions, initially set to False
    mask = np.full((dims[args.xdim], dims[args.ydim]), False)

    # Set the args.boundaries grid-cells closest to each boundary to True
    mask[: args.boundaries, :] = True  # top boundary
    mask[-args.boundaries :, :] = True  # bottom boundary
    mask[:, : args.boundaries] = True  # left boundary
    mask[:, -args.boundaries :] = True  # right boundary

    # Save the numpy array to a .npy file
    np.save(outdir + "border_mask", mask)  # (N_x, N_y)


if __name__ == "__main__":
    main()
