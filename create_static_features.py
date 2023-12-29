from argparse import ArgumentParser

import numpy as np
import xarray as xr

from neural_lam import constants


def main():
    parser = ArgumentParser(description='Static features arguments')
    parser.add_argument('--xdim', type=str, default="x_1",
                        help='Name of the x-dimension in the dataset (default: x_1)')
    parser.add_argument('--ydim', type=str, default="y_1",
                        help='Name of the x-dimension in the dataset (default: y_1)')
    parser.add_argument('--zdim', type=str, default="z_1",
                        help='Name of the x-dimension in the dataset (default: z_1)')
    parser.add_argument(
        '--field_names', nargs="+", default=["hsurf", "FI", "P0FL"],
        help='Names of the fields to extract from the .nc file (default: ["hsurf", "FI", "P0FL"])'),
    parser.add_argument(
        '--boundaries', type=int, default=30,
        help='Number of grid-cells closest to each boundary to mask (default: 30)')
    parser.add_argument(
        '--outdir', type=str, default="data/cosmo/static/",
        help='Output directory for the static features (default: data/cosmo/static/)')
    args = parser.parse_args()

    # Open the .nc file
    ds = xr.open_zarr(constants.example_file).isel(time=0)

    np_fields = []

    for var_name in args.field_names:
        # scale the variable to [0, 1]
        ds[var_name] = (ds[var_name] - ds[var_name].min()
                        ) / (ds[var_name].max() - ds[var_name].min())

        if args.zdim not in ds[var_name].dims:
            field_2d = ds[var_name].transpose(args.xdim, args.ydim).values
            # add a dummy dimension
            np_fields.append(np.expand_dims(field_2d, axis=-1))
        else:
            np_fields.append(ds[var_name].sel({args.zdim: constants.vertical_levels}).transpose(
                args.xdim, args.ydim, args.zdim).values)

    np_fields = np.concatenate(np_fields, axis=-1)

    # Save the numpy array to a .npy file
    np.save(args.outdir + 'reference_geopotential_pressure.npy', np_fields)

    # Get the dimensions of the dataset
    dims = ds.dims
    x_dim, y_dim = ds.dims[args.xdim], ds.dims[args.ydim]

    # Create a 2D meshgrid for x and y indices
    x_grid, y_grid = np.indices((x_dim, y_dim))

    # Stack the 2D arrays into a 3D array with x and y as the first dimension
    grid_xy = np.stack((y_grid, x_grid))

    np.save(args.outdir + 'nwp_xy.npy', grid_xy)

    # Create a mask with the same dimensions, initially set to False
    mask = np.full((dims[args.xdim], dims[args.ydim]), False)

    # Set the args.boundaries grid-cells closest to each boundary to True
    mask[:args.boundaries, :] = True  # top boundary
    mask[-args.boundaries:, :] = True  # bottom boundary
    mask[:, :args.boundaries] = True  # left boundary
    mask[:, -args.boundaries:] = True  # right boundary

    # Save the numpy array to a .npy file
    np.save(args.outdir + 'border_mask', mask)


if __name__ == "__main__":
    main()
