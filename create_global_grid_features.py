# Standard library
import os
from argparse import ArgumentParser

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr

# First-party
from neural_lam import vis

FIELD_NAMES = (
    "cos(lat)",
    "sin(lon)",
    "cos(lon)",
    "geopotential",
    "land-sea-mask",
)


def main():
    """
    Pre-compute all static features related to the grid nodes
    """
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="global_example_era5",
        help="Dataset to create grid features for "
        "(default: global_example_era5)",
    )
    parser.add_argument(
        "--plot",
        type=int,
        default=0,
        help="If fields should be plotted " "(default: 0 (false))",
    )
    args = parser.parse_args()

    static_dir_path = os.path.join("data", args.dataset, "static")
    fields_group_path = os.path.join("data", args.dataset, "fields.zarr")
    fields_group = zarr.open(fields_group_path, mode="r")

    grid_features_list = []  # Each (num_lon, num_lat) numpy array

    # Lat-lon
    grid_lat_vals = np.array(
        fields_group["latitude"], dtype=np.float32
    )  # (num_lat,)
    grid_lon_vals = np.array(
        fields_group["longitude"], dtype=np.float32
    )  # (num_long,)

    grid_lat = np.expand_dims(grid_lat_vals, axis=0).repeat(
        grid_lon_vals.shape[0], axis=0
    )  # (num_lon, num_lat)
    grid_lon = np.expand_dims(grid_lon_vals, axis=1).repeat(
        grid_lat_vals.shape[0], axis=1
    )  # (num_lon, num_lat)
    grid_lat_rad = np.deg2rad(grid_lat)
    grid_lon_rad = np.deg2rad(grid_lon)

    grid_features_list.append(np.cos(grid_lat_rad))
    grid_features_list.append(np.sin(grid_lon_rad))
    grid_features_list.append(np.cos(grid_lon_rad))

    # Geopotential
    geopotential_raw = np.array(
        fields_group["geopotential_at_surface"], dtype=np.float32
    )  # (num_lon, num_lat)
    gp_min = geopotential_raw.min()
    gp_max = geopotential_raw.max()
    # Rescale geopotential to [0,1]
    geopotential = (geopotential_raw - gp_min) / (
        gp_max - gp_min
    )  # (num_lon, num_lat)
    grid_features_list.append(geopotential)

    # Land-sea-mask
    land_sea_mask = np.array(
        fields_group["land_sea_mask"], dtype=np.float32
    )  # (num_lon, num_lat)
    grid_features_list.append(land_sea_mask)

    # Reshape and convert to torch
    grid_features_stacked = np.stack(
        grid_features_list, axis=2
    )  # (num_lon, num_lat, num_features)
    grid_features_np = grid_features_stacked.reshape(
        -1,
        grid_features_stacked.shape[-1],
    )  # Flatten first two dims, (num_grid_nodes,num_features, )
    grid_features = torch.tensor(grid_features_np, dtype=torch.float32)

    if args.plot:
        for feature, field_name in zip(grid_features.T, FIELD_NAMES):
            vis.plot_prediction(feature, feature, title=field_name)
            plt.show()

    torch.save(grid_features, os.path.join(static_dir_path, "grid_features.pt"))


if __name__ == "__main__":
    main()
