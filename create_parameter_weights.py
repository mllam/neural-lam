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
from neural_lam.weather_dataset import WeatherDataModule


def get_rank():
    """Get the rank of the current process in the distributed group."""
    return int(os.environ["SLURM_PROCID"])


def get_world_size():
    """Get the number of processes in the distributed group."""
    return int(os.environ["SLURM_NTASKS"])


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
    parser.add_argument("--dataset", type=str, default="meps_example")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--subset", type=int, default=8760)
    parser.add_argument("--n_workers", type=int, default=4)
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

    data_module = WeatherDataModule(
        dataset_name=args.dataset,
        standardize=False,
        subset=args.subset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
    )
    data_module.setup(stage="fit")

    train_sampler = DistributedSampler(
        data_module.train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        data_module.train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.n_workers,
    )

    if rank == 0:
        w_list = [
            pw * lw
            for var_name, pw in zip(
                constants.PARAM_NAMES_SHORT, constants.PARAM_WEIGHTS.values()
            )
            for lw in (
                constants.LEVEL_WEIGHTS.values()
                if constants.IS_3D[var_name]
                else [1]
            )
        ]
        np.save(
            os.path.join(static_dir_path, "parameter_weights.npy"),
            np.array(w_list, dtype="float32"),
        )

    means = []
    squares = []
    flux_means = []
    flux_squares = []
    for init_batch, target_batch, _, forcing_batch in tqdm(
        train_loader, disable=rank != 0
    ):
        batch = torch.cat((init_batch, target_batch), dim=1).to(device)
        means.append(torch.mean(batch, dim=(1, 2)))
        squares.append(torch.mean(batch**2, dim=(1, 2)))
        if constants.GRID_FORCING_DIM > 0:
            flux_batch = forcing_batch[:, :, :, 1].to(device)
            flux_means.append(torch.mean(flux_batch))
            flux_squares.append(torch.mean(flux_batch**2))

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

        if constants.GRID_FORCING_DIM > 0:
            flux_means_all = torch.stack(flux_means)
            flux_squares_all = torch.stack(flux_squares)
            flux_mean = torch.mean(flux_means_all)
            flux_second_moment = torch.mean(flux_squares_all)
            flux_std = torch.sqrt(flux_second_moment - flux_mean**2)
            torch.save(
                {"mean": flux_mean, "std": flux_std},
                os.path.join(static_dir_path, "flux_stats.pt"),
            )

    data_module = WeatherDataModule(
        dataset_name=args.dataset,
        standardize=True,
        subset=args.subset,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
    )
    data_module.setup(stage="fit")

    train_sampler = DistributedSampler(
        data_module.train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        data_module.train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.n_workers,
    )

    # Compute mean and std-dev of one-step differences
    diff_means = []
    diff_squares = []
    for init_batch, target_batch, _, _ in tqdm(train_loader, disable=rank != 0):
        batch = torch.cat((init_batch, target_batch), dim=1).to(device)
        diffs = batch[:, 1:] - batch[:, :-1]
        diff_means.append(torch.mean(diffs, dim=(1, 2)))
        diff_squares.append(torch.mean(diffs**2, dim=(1, 2)))

    dist.barrier()

    diff_means_gathered = [None] * world_size
    diff_squares_gathered = [None] * world_size
    dist.all_gather_object(diff_means_gathered, torch.cat(diff_means, dim=0))
    dist.all_gather_object(
        diff_squares_gathered, torch.cat(diff_squares, dim=0)
    )

    if rank == 0:
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
