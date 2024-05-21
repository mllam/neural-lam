# Standard library
import os
from argparse import ArgumentParser

# Third-party
import numpy as np
import torch
from tqdm import tqdm

# First-party
from neural_lam import config
from neural_lam.weather_dataset import WeatherDataset


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
        "--batch_size",
        type=int,
        default=32,
        help="Batch size when iterating over the dataset",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=3,
        help="Step length in hours to consider single time step (default: 3)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    args = parser.parse_args()

    config_loader = config.Config.from_file(args.data_config)
    static_dir_path = os.path.join("data", config_loader.dataset.name, "static")

    # Create parameter weights based on height
    # based on fig A.1 in graph cast paper
    w_dict = {
        "2": 1.0,
        "0": 0.1,
        "65": 0.065,
        "1000": 0.1,
        "850": 0.05,
        "500": 0.03,
    }
    w_list = np.array(
        [
            w_dict[par.split("_")[-2]]
            for par in config_loader.dataset.var_longnames
        ]
    )
    print("Saving parameter weights...")
    np.save(
        os.path.join(static_dir_path, "parameter_weights.npy"),
        w_list.astype("float32"),
    )

    # Load dataset without any subsampling
    ds = WeatherDataset(
        config_loader.dataset.name,
        split="train",
        subsample_step=1,
        pred_length=63,
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
    for init_batch, target_batch, forcing_batch in tqdm(loader):
        batch = torch.cat(
            (init_batch, target_batch), dim=1
        )  # (N_batch, N_t, N_grid, d_features)
        means.append(torch.mean(batch, dim=(1, 2)))  # (N_batch, d_features,)
        squares.append(
            torch.mean(batch**2, dim=(1, 2))
        )  # (N_batch, d_features,)

        # Flux at 1st windowed position is index 1 in forcing
        flux_batch = forcing_batch[:, :, :, 1]
        flux_means.append(torch.mean(flux_batch))  # (,)
        flux_squares.append(torch.mean(flux_batch**2))  # (,)

    mean = torch.mean(torch.cat(means, dim=0), dim=0)  # (d_features)
    second_moment = torch.mean(torch.cat(squares, dim=0), dim=0)
    std = torch.sqrt(second_moment - mean**2)  # (d_features)

    flux_mean = torch.mean(torch.stack(flux_means))  # (,)
    flux_second_moment = torch.mean(torch.stack(flux_squares))  # (,)
    flux_std = torch.sqrt(flux_second_moment - flux_mean**2)  # (,)
    flux_stats = torch.stack((flux_mean, flux_std))

    print("Saving mean, std.-dev, flux_stats...")
    torch.save(mean, os.path.join(static_dir_path, "parameter_mean.pt"))
    torch.save(std, os.path.join(static_dir_path, "parameter_std.pt"))
    torch.save(flux_stats, os.path.join(static_dir_path, "flux_stats.pt"))

    # Compute mean and std.-dev. of one-step differences across the dataset
    print("Computing mean and std.-dev. for one-step differences...")
    ds_standard = WeatherDataset(
        config_loader.dataset.name,
        split="train",
        subsample_step=1,
        pred_length=63,
        standardize=True,
    )  # Re-load with standardization
    loader_standard = torch.utils.data.DataLoader(
        ds_standard, args.batch_size, shuffle=False, num_workers=args.n_workers
    )
    used_subsample_len = (65 // args.step_length) * args.step_length

    diff_means = []
    diff_squares = []
    for init_batch, target_batch, _ in tqdm(loader_standard):
        batch = torch.cat(
            (init_batch, target_batch), dim=1
        )  # (N_batch, N_t', N_grid, d_features)
        # Note: batch contains only 1h-steps
        stepped_batch = torch.cat(
            [
                batch[:, ss_i : used_subsample_len : args.step_length]
                for ss_i in range(args.step_length)
            ],
            dim=0,
        )
        # (N_batch', N_t, N_grid, d_features),
        # N_batch' = args.step_length*N_batch

        batch_diffs = stepped_batch[:, 1:] - stepped_batch[:, :-1]
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
