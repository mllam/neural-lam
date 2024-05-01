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
from neural_lam import constants
from neural_lam.weather_dataset import WeatherDataset


def get_rank():
    """Get the rank of the current process in the distributed group."""
    if "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    parser = ArgumentParser()
    parser.add_argument(
        "--rank", type=int, default=0, help="Rank of the current process"
    )
    args, _ = parser.parse_known_args()
    return args.rank


def get_world_size():
    """Get the number of processes in the distributed group."""
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"])
    parser = ArgumentParser()
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of processes in the distributed group",
    )
    args, _ = parser.parse_known_args()
    return args.world_size


def setup(rank, world_size):  # pylint: disable=redefined-outer-name
    """Initialize the distributed group."""
    try:
        master_node = (
            subprocess.check_output(
                "scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1",
                shell=True,
            )
            .strip()
            .decode("utf-8")
        )
    except Exception as e:
        print(f"Error getting master node IP: {e}")
        raise
    master_port = "12355"
    os.environ["MASTER_ADDR"] = master_node
    os.environ["MASTER_PORT"] = master_port
    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    """Destroy the distributed group."""
    dist.destroy_process_group()


def main(rank, world_size):  # pylint: disable=redefined-outer-name
    """Compute the mean and standard deviation of the input data."""
    setup(rank, world_size)
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

    if args.subset % (world_size * args.batch_size) != 0:
        raise ValueError(
            "Subset size must be divisible by (world_size * batch_size)"
        )

    device = torch.device(
        f"cuda:{rank % torch.cuda.device_count()}"
        if torch.cuda.is_available()
        else "cpu"
    )
    static_dir_path = os.path.join("data", args.dataset, "static")

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
        [w_dict[par.split("_")[-2]] for par in constants.PARAM_NAMES]
    )
    print("Saving parameter weights...")
    np.save(
        os.path.join(static_dir_path, "parameter_weights.npy"),
        w_list.astype("float32"),
    )

    # Load dataset without any subsampling
    ds = WeatherDataset(
        args.dataset,
        split="train",
        subsample_step=1,
        pred_length=63,
        standardize=False,
    )  # Without standardization

    train_sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank)
    loader = torch.utils.data.DataLoader(
        ds,
        args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        sampler=train_sampler,
    )
    # Compute mean and std.-dev. of each parameter (+ flux forcing)
    # across full dataset
    print("Computing mean and std.-dev. for parameters...")
    means = []
    squares = []
    flux_means = []
    flux_squares = []
    for init_batch, target_batch, forcing_batch in tqdm(loader):
        batch = torch.cat((init_batch, target_batch), dim=1).to(
            device
        )  # (N_batch, N_t, N_grid, d_features)
        means.append(torch.mean(batch, dim=(1, 2)))  # (N_batch, d_features,)
        squares.append(
            torch.mean(batch**2, dim=(1, 2))
        )  # (N_batch, d_features,)

        # Flux at 1st windowed position is index 1 in forcing
        flux_batch = forcing_batch[:, :, :, 1]
        flux_means.append(torch.mean(flux_batch))  # (,)
        flux_squares.append(torch.mean(flux_batch**2))  # (,)
    dist.barrier()

    means_gathered = [None] * world_size
    squares_gathered = [None] * world_size
    dist.all_gather_object(means_gathered, torch.cat(means, dim=0))
    dist.all_gather_object(squares_gathered, torch.cat(squares, dim=0))

    if rank == 0:
        means_all = torch.cat(means_gathered, dim=0)
        squares_all = torch.cat(squares_gathered, dim=0)
        mean = torch.mean(means_all, dim=0)
        second_moment = torch.mean(squares_all, dim=0)
        std = torch.sqrt(second_moment - mean**2)
        torch.save(mean, os.path.join(static_dir_path, "parameter_mean.pt"))
        torch.save(std, os.path.join(static_dir_path, "parameter_std.pt"))

        flux_means_all = torch.stack(flux_means)
        flux_squares_all = torch.stack(flux_squares)
        flux_mean = torch.mean(flux_means_all)
        flux_second_moment = torch.mean(flux_squares_all)
        flux_std = torch.sqrt(flux_second_moment - flux_mean**2)
        torch.save(
            {"mean": flux_mean, "std": flux_std},
            os.path.join(static_dir_path, "flux_stats.pt"),
        )
    # Compute mean and std.-dev. of one-step differences across the dataset
    print("Computing mean and std.-dev. for one-step differences...")
    ds_standard = WeatherDataset(
        args.dataset,
        split="train",
        subsample_step=1,
        pred_length=63,
        standardize=True,
    )  # Re-load with standardization
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

    for init_batch, target_batch, _, _ in tqdm(
        loader_standard, disable=rank != 0
    ):
        batch = torch.cat((init_batch, target_batch), dim=1).to(device)
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

    dist.barrier()

    diff_means_gathered = [None] * world_size
    diff_squares_gathered = [None] * world_size
    dist.all_gather_object(diff_means_gathered, torch.cat(diff_means, dim=0))
    dist.all_gather_object(
        diff_squares_gathered, torch.cat(diff_squares, dim=0)
    )
    diff_means_all = torch.cat(diff_means_gathered, dim=0)
    diff_squares_all = torch.cat(diff_squares_gathered, dim=0)
    diff_mean = torch.mean(diff_means_all, dim=0)
    diff_second_moment = torch.mean(diff_squares_all, dim=0)
    diff_std = torch.sqrt(diff_second_moment - diff_mean**2)
    torch.save(diff_mean, os.path.join(static_dir_path, "diff_mean.pt"))
    torch.save(diff_std, os.path.join(static_dir_path, "diff_std.pt"))

    cleanup()


if __name__ == "__main__":
    rank = get_rank()
    world_size = get_world_size()
    main(rank, world_size)
