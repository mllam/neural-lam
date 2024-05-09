# Standard library
from argparse import ArgumentParser

# Third-party
import torch
import xarray as xr
from tqdm import tqdm

# First-party
from neural_lam.weather_dataset import WeatherDataModule


def main():
    """
    Pre-compute parameter weights to be used in loss function
    """
    parser = ArgumentParser(description="Training arguments")
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

    data_module = WeatherDataModule(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    data_module.setup()
    loader = data_module.train_dataloader()

    # Load dataset without any subsampling
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
