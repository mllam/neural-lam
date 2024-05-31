# Standard library
import os
from argparse import ArgumentParser

# Third-party
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

# First-party
from neural_lam import config
from neural_lam.weather_dataset import WeatherDataModule, WeatherDataset


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
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--zarr_path",
        type=str,
        default="normalization.zarr",
        help="Directory where data is stored",
    )

    args = parser.parse_args()

    config_loader = config.Config.from_file(args.data_config)
    static_dir_path = os.path.join(
        "data", config_loader.dataset.name, "static"
    )

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
    data_module = WeatherDataModule(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    data_module.setup()
    loader = data_module.train_dataloader()

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
    # Compute mean and std.-dev. of each parameter (+ forcing forcing)
    # across full dataset
    print("Computing mean and std.-dev. for parameters...")
    means = []
    squares = []
    fb_means = {"forcing": [], "boundary": []}
    fb_squares = {"forcing": [], "boundary": []}

    for init_batch, target_batch, forcing_batch, boundary_batch, _ in tqdm(
        loader
    ):
        batch = torch.cat(
            (init_batch, target_batch), dim=1
        )  # (N_batch, N_t, N_grid, d_features)
        means.append(torch.mean(batch, dim=(1, 2)))  # (N_batch, d_features,)
        squares.append(torch.mean(batch**2, dim=(1, 2)))

        for fb_type, fb_batch in zip(
            ["forcing", "boundary"], [forcing_batch, boundary_batch]
        ):
            fb_batch = fb_batch[:, :, :, 1]
            fb_means[fb_type].append(torch.mean(fb_batch))  # (,)
            fb_squares[fb_type].append(torch.mean(fb_batch**2))  # (,)

    mean = torch.mean(torch.cat(means, dim=0), dim=0)  # (d_features)
    second_moment = torch.mean(torch.cat(squares, dim=0), dim=0)
    std = torch.sqrt(second_moment - mean**2)  # (d_features)

    fb_stats = {}
    for fb_type in ["forcing", "boundary"]:
        fb_stats[f"{fb_type}_mean"] = torch.mean(
            torch.stack(fb_means[fb_type])
        )  # (,)
        fb_second_moment = torch.mean(torch.stack(fb_squares[fb_type]))  # (,)
        fb_stats[f"{fb_type}_std"] = torch.sqrt(
            fb_second_moment - fb_stats[f"{fb_type}_mean"] ** 2
        )  # (,)

    # Compute mean and std.-dev. of one-step differences across the dataset
    print("Computing mean and std.-dev. for one-step differences...")
    diff_means = []
    diff_squares = []
    for init_batch, target_batch, _, _, _ in tqdm(loader):
        # normalize the batch
        init_batch = (init_batch - mean) / std
        target_batch = (target_batch - mean) / std

        batch = torch.cat((init_batch, target_batch), dim=1)
        batch_diffs = batch[:, 1:] - batch[:, :-1]
        # (N_batch, N_t-1, N_grid, d_features)

        diff_means.append(
            torch.mean(batch_diffs, dim=(1, 2))
        )  # (N_batch', d_features,)
        diff_squares.append(
            torch.mean(batch_diffs**2, dim=(1, 2))
        )  # (N_batch', d_features,)

    diff_mean = torch.mean(torch.cat(diff_means, dim=0), dim=0)  # (d_features)
    diff_second_moment = torch.mean(torch.cat(diff_squares, dim=0), dim=0)
    diff_std = torch.sqrt(diff_second_moment - diff_mean**2)  # (d_features)

    # Create xarray dataset
    ds = xr.Dataset(
        {
            "mean": (["d_features"], mean),
            "std": (["d_features"], std),
            "diff_mean": (["d_features"], diff_mean),
            "diff_std": (["d_features"], diff_std),
            **fb_stats,
        }
    )

    # Save dataset as Zarr
    print("Saving dataset as Zarr...")
    ds.to_zarr(args.zarr_path, mode="w")


if __name__ == "__main__":
    main()
