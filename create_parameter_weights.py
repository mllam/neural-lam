# Standard library
import os
import subprocess
from argparse import ArgumentParser

# Third-party
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# First-party
from neural_lam import config
from neural_lam.weather_dataset import WeatherDataset


class PaddedWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, world_size, batch_size):
        super().__init__()
        self.base_dataset = base_dataset
        self.world_size = world_size
        self.batch_size = batch_size
        self.total_samples = len(base_dataset)
        self.padded_samples = (
            (self.world_size * self.batch_size) - self.total_samples
        ) % self.world_size
        self.original_indices = list(range(self.total_samples))
        self.padded_indices = list(
            range(self.total_samples, self.total_samples + self.padded_samples)
        )

    def __getitem__(self, idx):
        if idx >= self.total_samples:
            # Return a padded item (zeros or a repeat of the last item)
            # Repeat last item
            return self.base_dataset[self.original_indices[-1]]
        return self.base_dataset[idx]

    def __len__(self):
        return self.total_samples + self.padded_samples

    def get_original_indices(self):
        return self.original_indices


def get_rank():
    """Get the rank of the current process in the distributed group."""
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    return 0


def get_world_size():
    """Get the number of processes in the distributed group."""
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"])
    return 1


def setup(rank, world_size):  # pylint: disable=redefined-outer-name
    """Initialize the distributed group."""
    if "SLURM_JOB_NODELIST" in os.environ:
        master_node = (
            subprocess.check_output(
                "scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1",
                shell=True,
            )
            .strip()
            .decode("utf-8")
        )
    else:
        master_node = "localhost"
    master_port = "12355"
    os.environ["MASTER_ADDR"] = master_node
    os.environ["MASTER_PORT"] = master_port
    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(
        f"Initialized {dist.get_backend()} process group with "
        f"world size "
        f"{world_size}."
    )


def main(rank, world_size):  # pylint: disable=redefined-outer-name
    """Compute the mean and standard deviation of the input data."""
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

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if rank == 0:
        static_dir_path = os.path.join(
            "data", config_loader.dataset.name, "static"
        )

        # Create parameter weights based on height
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
    )
    ds = PaddedWeatherDataset(ds, world_size, args.batch_size)

    train_sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank)
    loader = torch.utils.data.DataLoader(
        ds,
        args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        sampler=train_sampler,
    )
    # Compute mean and std.-dev. of each parameter (+ flux forcing) across
    # full dataset
    if rank == 0:
        print("Computing mean and std.-dev. for parameters...")
    means = []
    squares = []
    flux_means = []
    flux_squares = []

    for i in range(100):
        # Data loading and initial computations remain on GPU
        for init_batch, target_batch, forcing_batch in tqdm(loader):
            init_batch, target_batch, forcing_batch = (
                init_batch.to(device),
                target_batch.to(device),
                forcing_batch.to(device),
            )
            batch = torch.cat((init_batch, target_batch), dim=1)
            # Move to CPU after computation
            means.append(torch.mean(batch, dim=(1, 2)).cpu())
            # Move to CPU after computation
            squares.append(torch.mean(batch**2, dim=(1, 2)).cpu())
            flux_batch = forcing_batch[:, :, :, 1]
            # Move to CPU after computation
            flux_means.append(torch.mean(flux_batch).cpu())
            # Move to CPU after computation
            flux_squares.append(torch.mean(flux_batch**2).cpu())

    means_gathered = [None] * world_size
    squares_gathered = [None] * world_size
    # Aggregation remains unchanged but ensures inputs are on CPU
    dist.all_gather_object(means_gathered, torch.cat(means, dim=0))
    dist.all_gather_object(squares_gathered, torch.cat(squares, dim=0))

    if rank == 0:
        # Final computations and saving are done on CPU
        means_all = torch.cat(means_gathered, dim=0)
        squares_all = torch.cat(squares_gathered, dim=0)
        original_indices = ds.get_original_indices()
        means_filtered = [means_all[i] for i in original_indices]
        squares_filtered = [squares_all[i] for i in original_indices]
        mean = torch.mean(torch.stack(means_filtered), dim=0)
        second_moment = torch.mean(torch.stack(squares_filtered), dim=0)
        std = torch.sqrt(second_moment - mean**2)
        torch.save(
            mean.cpu(), os.path.join(static_dir_path, "parameter_mean.pt")
        )  # Ensure tensor is on CPU
        torch.save(
            std.cpu(), os.path.join(static_dir_path, "parameter_std.pt")
        )  # Ensure tensor is on CPU

        # flux_means_filtered = [flux_means[i] for i in original_indices]
        # flux_squares_filtered = [flux_squares[i] for i in original_indices]
        flux_means_all = torch.stack(flux_means)
        flux_squares_all = torch.stack(flux_squares)
        flux_mean = torch.mean(flux_means_all)
        flux_second_moment = torch.mean(flux_squares_all)
        flux_std = torch.sqrt(flux_second_moment - flux_mean**2)
        torch.save(
            torch.stack((flux_mean, flux_std)).cpu(),
            os.path.join(static_dir_path, "flux_stats.pt"),
        )  # Ensure tensor is on CPU
    # Compute mean and std.-dev. of one-step differences across the dataset
    dist.barrier()
    if rank == 0:
        print("Computing mean and std.-dev. for one-step differences...")
    ds_standard = WeatherDataset(
        config_loader.dataset.name,
        split="train",
        subsample_step=1,
        pred_length=63,
        standardize=True,
    )
    ds_standard = PaddedWeatherDataset(ds_standard, world_size, args.batch_size)

    sampler_standard = DistributedSampler(
        ds_standard, num_replicas=world_size, rank=rank
    )
    loader_standard = torch.utils.data.DataLoader(
        ds_standard,
        args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        sampler=sampler_standard,
    )
    used_subsample_len = (65 // args.step_length) * args.step_length

    diff_means = []
    diff_squares = []

    for init_batch, target_batch, _ in tqdm(loader_standard, disable=rank != 0):
        init_batch, target_batch = init_batch.to(device), target_batch.to(
            device
        )
        batch = torch.cat((init_batch, target_batch), dim=1)
        stepped_batch = torch.cat(
            [
                batch[:, ss_i : used_subsample_len : args.step_length]
                for ss_i in range(args.step_length)
            ],
            dim=0,
        )
        batch_diffs = stepped_batch[:, 1:] - stepped_batch[:, :-1]

        # Compute means and squares on GPU, then move to CPU for storage
        diff_means.append(torch.mean(batch_diffs, dim=(1, 2)).cpu())
        diff_squares.append(torch.mean(batch_diffs**2, dim=(1, 2)).cpu())

    dist.barrier()

    diff_means_gathered = [None] * world_size
    diff_squares_gathered = [None] * world_size
    dist.all_gather_object(diff_means_gathered, torch.cat(diff_means, dim=0))
    dist.all_gather_object(
        diff_squares_gathered, torch.cat(diff_squares, dim=0)
    )

    if rank == 0:
        # Concatenate and compute final statistics on CPU
        diff_means_all = torch.cat(diff_means_gathered, dim=0)
        diff_squares_all = torch.cat(diff_squares_gathered, dim=0)
        original_indices = ds_standard.get_original_indices()
        diff_means_filtered = [diff_means_all[i] for i in original_indices]
        diff_squares_filtered = [diff_squares_all[i] for i in original_indices]
        diff_mean = torch.mean(torch.stack(diff_means_filtered), dim=0)
        diff_second_moment = torch.mean(
            torch.stack(diff_squares_filtered), dim=0
        )
        diff_std = torch.sqrt(diff_second_moment - diff_mean**2)

        # Save tensors to disk, ensuring they are on CPU
        torch.save(diff_mean, os.path.join(static_dir_path, "diff_mean.pt"))
        torch.save(diff_std, os.path.join(static_dir_path, "diff_std.pt"))

    dist.destroy_process_group()


if __name__ == "__main__":
    rank = get_rank()
    world_size = get_world_size()
    setup(rank, world_size)
    main(rank, world_size)
