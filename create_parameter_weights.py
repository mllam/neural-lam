# Standard library
import os
from argparse import ArgumentParser

# Third-party
import numpy as np
import torch
from tqdm import tqdm

# First-party
from neural_lam import constants
from neural_lam.weather_dataset import WeatherDataset


def main():
    """
    Pre-compute parameter weights to be used in loss function
    """
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="meps_example",
        help="Dataset to compute weights for (default: meps_example)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size when iterating over the dataset",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=1,
        help="Step length in hours to consider single time step (default: 1)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    args = parser.parse_args()

    static_dir_path = os.path.join("data", args.dataset, "static")

    # Create parameter weights based on height
    w_list = []
    for var_name, pw in zip(
        constants.PARAM_NAMES_SHORT, constants.PARAM_WEIGHTS.values()
    ):
        # Determine the levels to iterate over
        levels = (
            constants.LEVEL_WEIGHTS.values()
            if constants.IS_3D[var_name]
            else [1]
        )

        # Iterate over the levels
        for lw in levels:
            w_list.append(pw * lw)

    w_list = np.array(w_list)

    print("Saving parameter weights...")
    np.save(
        os.path.join(static_dir_path, "parameter_weights.npy"),
        w_list.astype("float32"),
    )

    # Load dataset without any subsampling
    ds = WeatherDataset(
        args.dataset,
        split="train",
        standardize=False,
    )  # Without standardization
    loader = torch.utils.data.DataLoader(
        ds, args.batch_size, shuffle=False, num_workers=args.n_workers
    )
    # Compute mean and std.-dev. of each parameter (+ flux forcing)
    # across full dataset
    print("Computing mean and std.-dev. for parameters...")

    means = []
    squares = []
    flux_means = []
    flux_squares = []
    for batch_data in tqdm(loader):
        if constants.GRID_FORCING_DIM > 0:
            init_batch, target_batch, _, forcing_batch = batch_data
            flux_batch = forcing_batch[
                :, :, :, :3
            ]  # fluxes are first 3 features
            flux_means.append(torch.mean(flux_batch, dim=(1, 2, 3)))  # (,)
            flux_squares.append(torch.mean(flux_batch**2, dim=(1, 2, 3)))  # (,)
        else:
            init_batch, target_batch, _ = batch_data

        batch = torch.cat(
            (init_batch, target_batch), dim=1
        )  # (N_batch, N_t, N_grid, d_features)
        means.append(torch.mean(batch, dim=(1, 2)))  # (N_batch, d_features,)
        squares.append(
            torch.mean(batch**2, dim=(1, 2))
        )  # (N_batch, d_features,)

    mean = torch.mean(torch.cat(means, dim=0), dim=0)  # (d_features)
    second_moment = torch.mean(torch.cat(squares, dim=0), dim=0)
    std = torch.sqrt(second_moment - mean**2)  # (d_features)

    if constants.GRID_FORCING_DIM > 0:
        flux_mean = torch.mean(torch.cat(flux_means, dim=0), dim=0)  # (,)
        flux_second_moment = torch.mean(
            torch.cat(flux_squares, dim=0), dim=0
        )  # (,)
        flux_std = torch.sqrt(flux_second_moment - flux_mean**2)  # (,)
        flux_stats = torch.stack((flux_mean, flux_std))

        print("Saving mean flux_stats...")
        torch.save(flux_stats, os.path.join(static_dir_path, "flux_stats.pt"))
    print("Saving mean, std.-dev...")
    torch.save(mean, os.path.join(static_dir_path, "parameter_mean.pt"))
    torch.save(std, os.path.join(static_dir_path, "parameter_std.pt"))

    # Compute mean and std.-dev. of one-step differences across the dataset
    print("Computing mean and std.-dev. for one-step differences...")
    ds_standard = WeatherDataset(
        args.dataset,
        split="train",
        standardize=True,
    )  # Re-load with standardization
    loader_standard = torch.utils.data.DataLoader(
        ds_standard, args.batch_size, shuffle=False, num_workers=args.n_workers
    )

    diff_means = []
    diff_squares = []
    for batch_data in tqdm(loader_standard):
        if constants.GRID_FORCING_DIM > 0:
            init_batch, target_batch, _, forcing_batch = batch_data
            flux_batch = forcing_batch[
                :, :, :, :3
            ]  # fluxes are first 3 features
            flux_means.append(torch.mean(flux_batch, dim=(1, 2, 3)))  # (,)
            flux_squares.append(torch.mean(flux_batch**2, dim=(1, 2, 3)))  # (,)
        else:
            init_batch, target_batch, _ = batch_data
        batch_diffs = init_batch[:, 1:] - target_batch
        # (N_batch', N_t-1, N_grid, d_features)

        diff_means.append(
            torch.mean(batch_diffs, dim=(1, 2))
        )  # (N_batch', d_features,)
        diff_squares.append(
            torch.mean(batch_diffs**2, dim=(1, 2))
        )  # (N_batch', d_features,)

    diff_mean = torch.mean(torch.cat(diff_means, dim=0), dim=0)  # (d_features)
    diff_second_moment = torch.mean(torch.cat(diff_squares, dim=0), dim=0)
    diff_std = torch.sqrt(diff_second_moment - diff_mean**2)  # (d_features)

    print("Saving one-step difference mean and std.-dev...")
    torch.save(diff_mean, os.path.join(static_dir_path, "diff_mean.pt"))
    torch.save(diff_std, os.path.join(static_dir_path, "diff_std.pt"))


if __name__ == "__main__":
    main()
