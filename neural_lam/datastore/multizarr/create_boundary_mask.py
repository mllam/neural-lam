# Standard library
from argparse import ArgumentParser
from pathlib import Path

# Third-party
import numpy as np
import xarray as xr

# Local
from . import config

DEFAULT_FILENAME = "boundary_mask.zarr"


def create_boundary_mask(data_config_path, zarr_path, n_boundary_cells):
    """Create a mask for the boundaries of the grid.

    Parameters
    ----------
    data_config_path : str
        Data configuration.
    zarr_path : str
        Path to save the Zarr archive.
    """
    data_config_path = config.Config.from_file(str(data_config_path))
    mask = np.zeros(list(data_config_path.grid_shape_state.values.values()))

    # Set the n_boundary_cells grid-cells closest to each boundary to True
    mask[:n_boundary_cells, :] = True  # top boundary
    mask[-n_boundary_cells:, :] = True  # noqa bottom boundary
    mask[:, :n_boundary_cells] = True  # left boundary
    mask[:, -n_boundary_cells:] = True  # noqa right boundary

    mask = xr.Dataset({"mask": (["y", "x"], mask)})

    print(f"Saving mask to {zarr_path}...")
    mask.to_zarr(zarr_path, mode="w")


def main():
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "data_config",
        type=str,
        help="Path to data config file",
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default=None,
        help="Path to save the Zarr archive "
        "(default: same directory as data config)",
    )
    parser.add_argument(
        "--n_boundary_cells",
        type=int,
        default=30,
        help="Number of grid-cells to set to True along each boundary",
    )
    args = parser.parse_args()

    if args.zarr_path is None:
        args.zarr_path = Path(args.data_config).parent / DEFAULT_FILENAME
    else:
        zarr_path = Path(args.zarr_path)

    create_boundary_mask(
        data_config_path=args.data_config,
        zarr_path=zarr_path,
        n_boundary_cells=args.n_boundary_cells,
    )


if __name__ == "__main__":
    main()
