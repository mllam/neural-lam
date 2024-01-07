import os
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

from neural_lam import constants
from neural_lam.weather_dataset import WeatherDataset


def main():
    parser = ArgumentParser(description='Training arguments')
    parser.add_argument('--dataset', type=str, default="meps_example",
                        help='Dataset to compute weights for (default: meps_example)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size when iterating over the dataset')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of workers in data loader (default: 4)')
    args = parser.parse_args()

    static_dir_path = os.path.join("data", args.dataset, "static")

    # Define weights for each vertical level and parameter
    # Create parameter weights based on height
    w_list = []
    for var_name, pw in zip(constants.param_names_short,
                            constants.param_weights.values()):
        # Determine the levels to iterate over
        levels = constants.level_weights.values() if constants.is_3d[var_name] else [1]

        # Iterate over the levels
        for lw in levels:
            w_list.append(pw * lw)

    w_list = np.array(w_list)

    print("Saving parameter weights...")
    np.save(
        os.path.join(
            static_dir_path,
            'parameter_weights.npy'),
        w_list.astype('float32'))

    # Load dataset without any subsampling
    ds = WeatherDataset(
        args.dataset,
        split="train",
        standardize=False)  # Without standardization
    loader = torch.utils.data.DataLoader(ds, args.batch_size, shuffle=False,
                                         num_workers=args.n_workers)
    # Compute mean and std.-dev. of each parameter (+ flux forcing) across full dataset
    print("Computing mean and std.-dev. for parameters...")
    means = []
    squares = []
    for init_batch, target_batch in tqdm(loader):
        batch = torch.cat((init_batch, target_batch),
                          dim=1)  # (N_batch, N_t, N_grid, d_features)
        means.append(torch.mean(batch, dim=(1, 2)))  # (N_batch, d_features,)
        # (N_batch, d_features,)
        squares.append(torch.mean(batch**2, dim=(1, 2)))

    mean = torch.mean(torch.cat(means, dim=0), dim=0)  # (d_features)
    second_moment = torch.mean(torch.cat(squares, dim=0), dim=0)
    std = torch.sqrt(second_moment - mean**2)  # (d_features)

    print("Saving mean, std.-dev...")
    torch.save(mean, os.path.join(static_dir_path, "parameter_mean.pt"))
    torch.save(std, os.path.join(static_dir_path, "parameter_std.pt"))

    # Compute mean and std.-dev. of one-step differences across the dataset
    print("Computing mean and std.-dev. for one-step differences...")
    ds_standard = WeatherDataset(
        args.dataset,
        split="train",
        standardize=True)  # Re-load with standardization
    loader_standard = torch.utils.data.DataLoader(
        ds_standard, args.batch_size, shuffle=False, num_workers=args.n_workers)

    diff_means = []
    diff_squares = []
    for init_batch, target_batch, in tqdm(loader_standard):
        batch_diffs = init_batch[:, 1:] - target_batch
        # (N_batch', N_t-1, N_grid, d_features)

        diff_means.append(torch.mean(batch_diffs, dim=(1, 2))
                          )  # (N_batch', d_features,)
        diff_squares.append(torch.mean(batch_diffs**2,
                                       dim=(1, 2)))  # (N_batch', d_features,)

    diff_mean = torch.mean(torch.cat(diff_means, dim=0), dim=0)  # (d_features)
    diff_second_moment = torch.mean(torch.cat(diff_squares, dim=0), dim=0)
    diff_std = torch.sqrt(diff_second_moment - diff_mean**2)  # (d_features)

    print("Saving one-step difference mean and std.-dev...")
    torch.save(diff_mean, os.path.join(static_dir_path, "diff_mean.pt"))
    torch.save(diff_std, os.path.join(static_dir_path, "diff_std.pt"))


if __name__ == "__main__":
    main()
