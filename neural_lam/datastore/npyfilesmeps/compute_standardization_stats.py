"""Utilities for computing MEPS datastore standardization statistics."""

# Standard library
import os
import subprocess
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

# Third-party
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# First-party
from neural_lam import WeatherDataset
from neural_lam.datastore import init_datastore
from neural_lam.utils import get_integer_time


class PaddedWeatherDataset(torch.utils.data.Dataset):
    """Wrap :class:`WeatherDataset` to pad samples for distributed runners."""

    def __init__(self, base_dataset, world_size, batch_size):
        """
        Parameters
        ----------
        base_dataset : WeatherDataset
            Dataset to pad.
        world_size : int
            Total number of distributed ranks participating.
        batch_size : int
            Per-rank batch size.
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.world_size = world_size
        self.batch_size = batch_size
        self.total_samples = len(base_dataset)
        self.padded_samples = (
            (self.world_size * self.batch_size) - self.total_samples
        ) % self.world_size
        self.original_indices = list(range(len(base_dataset)))
        self.padded_indices = list(
            range(self.total_samples, self.total_samples + self.padded_samples)
        )

    def __getitem__(self, idx):
        """Return an item, repeating the final sample for padded indices."""
        return self.base_dataset[
            (
                self.original_indices[-1]
                if idx >= self.total_samples
                else idx % len(self.base_dataset)
            )
        ]

    def __len__(self):
        """Return the padded dataset length."""
        return self.total_samples + self.padded_samples

    def get_original_indices(self):
        """Return indices of the non-padded samples."""
        return self.original_indices


def get_rank():
    """
    Return the rank inferred from SLURM or default to 0.

    Returns
    -------
    int
        The current process rank.
    """
    return int(os.environ.get("SLURM_PROCID", 0))


def get_world_size():
    """
    Return the world size inferred from SLURM or default to 1.

    Returns
    -------
    int
        The total number of processes in the distributed group.
    """
    return int(os.environ.get("SLURM_NTASKS", 1))


def setup(rank, world_size):  # pylint: disable=redefined-outer-name
    """
    Initialize the distributed group.

    Parameters
    ----------
    rank : int
        The rank of the current process.
    world_size : int
        The total number of processes.

    Raises
    ------
    RuntimeError
        If ``SLURM_JOB_NODELIST`` is set but no hostnames can be retrieved.
    """
    if "SLURM_JOB_NODELIST" in os.environ:
        nodelist = os.environ["SLURM_JOB_NODELIST"]
        hostnames = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist],
        )
        hostname_lines = hostnames.decode("utf-8").splitlines()
        if not hostname_lines:
            raise RuntimeError(
                f"SLURM_JOB_NODELIST is set to {repr(nodelist)}, but "
                "'scontrol show hostnames' returned no hostnames. "
                "Please check your SLURM job configuration."
            )
        master_node = hostname_lines[0].strip()
    else:
        print(
            "\033[91mCareful, you are running this script with --distributed "
            "without any scheduler. In most cases this will result in slower "
            "execution and the --distributed flag should be removed.\033[0m"
        )
        master_node = "localhost"
    os.environ["MASTER_ADDR"] = master_node
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(
        "nccl" if torch.cuda.is_available() else "gloo",
        rank=rank,
        world_size=world_size,
    )
    if rank == 0:
        print(
            f"Initialized {dist.get_backend()} "
            f"process group with world size {world_size}."
        )


def save_stats(
    static_dir_path, means, squares, flux_means, flux_squares, filename_prefix
):
    """
    Aggregate running statistics and persist them to ``static_dir_path``.

    Parameters
    ----------
    static_dir_path : str or pathlib.Path
        Directory where ``*.pt`` files should be written.
    means : Sequence[torch.Tensor]
        Shape ``(B, num_state_vars)``. Batch-wise means. Sequence of one or
        more tensors each of shape ``(B_i, num_state_vars)``; concatenated /
        stacked along the batch dim before reduction. Typical callers pass a
        single already-gathered tensor. The list can have length 0 to skip
        saving.
    squares : Sequence[torch.Tensor]
        Shape ``(B, num_state_vars)``. Batch-wise second moments. Sequence of
        one or more tensors each of shape ``(B_i, num_state_vars)``;
        concatenated / stacked along the batch dim before reduction. Typical
        callers pass a single already-gathered tensor. The list can have
        length 0 to skip saving.
    flux_means : Sequence[torch.Tensor]
        Shape ``(B,)``. Flux means. Sequence of one or more tensors each of
        shape ``(B_i,)``; concatenated / stacked along the batch dim before
        reduction. Typical callers pass a single already-gathered tensor.
        The list can have length 0 to skip saving.
    flux_squares : Sequence[torch.Tensor]
        Shape ``(B,)``. Flux second moments. Sequence of one or more tensors
        each of shape ``(B_i,)``; concatenated / stacked along the batch dim
        before reduction. Typical callers pass a single already-gathered
        tensor. The list can have length 0 to skip saving.
    filename_prefix : str
        Prefix (e.g., ``"parameter"`` or ``"diff"``) for saved tensors.
    """
    means = (
        torch.stack(means) if len(means) > 1 else means[0]
    )  # (B, d_features,)
    squares = (
        torch.stack(squares) if len(squares) > 1 else squares[0]
    )  # (B, d_features,)
    mean = torch.mean(means, dim=0)  # (d_features,)
    second_moment = torch.mean(squares, dim=0)  # (d_features,)
    std = torch.sqrt(second_moment - mean**2)  # (d_features,)
    print(
        f"Saving {filename_prefix} mean and std.-dev. to "
        f"{filename_prefix}_mean.pt and {filename_prefix}_std.pt"
    )
    torch.save(
        mean.cpu(), os.path.join(static_dir_path, f"{filename_prefix}_mean.pt")
    )
    torch.save(
        std.cpu(), os.path.join(static_dir_path, f"{filename_prefix}_std.pt")
    )

    if len(flux_means) == 0:
        return
    flux_means = (
        torch.stack(flux_means) if len(flux_means) > 1 else flux_means[0]
    )  # (B,)
    flux_squares = (
        torch.stack(flux_squares) if len(flux_squares) > 1 else flux_squares[0]
    )  # (B,)
    flux_mean = torch.mean(flux_means)  # (,)
    flux_second_moment = torch.mean(flux_squares)  # (,)
    flux_std = torch.sqrt(flux_second_moment - flux_mean**2)  # (,)
    print("Saving flux mean and std.-dev. to flux_stats.pt")
    torch.save(
        torch.stack((flux_mean, flux_std)).cpu(),
        os.path.join(static_dir_path, "flux_stats.pt"),
    )


def main(
    datastore_config_path, batch_size, step_length, n_workers, distributed
):
    """
    Pre-compute and persist standardization statistics from the datastore.

    Parameters
    ----------
    datastore_config_path : str or pathlib.Path
        Path to the MEPS datastore configuration file.
    batch_size : int
        Batch size used while iterating through the dataset.
    step_length : datetime.timedelta
        Temporal sampling interval for the difference statistics.
    n_workers : int
        Number of dataloader workers.
    distributed : bool
        If ``True``, run using torch.distributed with SLURM settings.
    """

    rank = get_rank()
    world_size = get_world_size()
    datastore = init_datastore(
        datastore_kind="npyfilesmeps", config_path=datastore_config_path
    )

    static_dir_path = Path(datastore_config_path).parent / "static"
    os.makedirs(static_dir_path, exist_ok=True)

    if distributed:
        setup(rank, world_size)
        device = torch.device(
            f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        )
        torch.cuda.set_device(device) if torch.cuda.is_available() else None

    # Setting this to the original value of the Oskarsson et al. paper (2023)
    # 65 forecast steps - 2 initial steps = 63
    ar_steps = 63
    # Raw (non-standardized) data for computing mean/std
    ds = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=ar_steps,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
    )
    if distributed:
        ds = PaddedWeatherDataset(
            ds,
            world_size,
            batch_size,
        )
        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        sampler = None
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size,
        shuffle=False,
        num_workers=n_workers,
        sampler=sampler,
    )

    if rank == 0:
        print("Computing mean and std.-dev. for parameters...")
    means, squares, flux_means, flux_squares = [], [], [], []

    for init_batch, target_batch, forcing_batch, _ in tqdm(loader):
        if distributed:
            init_batch, target_batch, forcing_batch = (
                init_batch.to(device),
                target_batch.to(device),
                forcing_batch.to(device),
            )
        # (B, N_t, num_grid_nodes, d_features)
        batch = torch.cat((init_batch, target_batch), dim=1)
        # Flux at 1st windowed position is index 0 in forcing
        flux_batch = forcing_batch[:, :, :, 0]
        # (B, d_features,)
        means.append(torch.mean(batch, dim=(1, 2)).cpu())
        squares.append(
            torch.mean(batch**2, dim=(1, 2)).cpu()
        )  # (B, d_features,)
        flux_means.append(torch.mean(flux_batch).cpu())  # (,)
        flux_squares.append(torch.mean(flux_batch**2).cpu())  # (,)

    if distributed and world_size > 1:
        means_gathered, squares_gathered = [None] * world_size, [
            None
        ] * world_size
        flux_means_gathered, flux_squares_gathered = (
            [None] * world_size,
            [None] * world_size,
        )
        dist.all_gather_object(means_gathered, torch.cat(means, dim=0))
        dist.all_gather_object(squares_gathered, torch.cat(squares, dim=0))
        dist.all_gather_object(flux_means_gathered, flux_means)
        dist.all_gather_object(flux_squares_gathered, flux_squares)

        if rank == 0:
            means_gathered, squares_gathered = (
                torch.cat(means_gathered, dim=0),
                torch.cat(squares_gathered, dim=0),
            )

            original_indices = ds.get_original_indices()
            means, squares = (
                [means_gathered[i] for i in original_indices],
                [squares_gathered[i] for i in original_indices],
            )
            flux_means = [
                torch.cat(
                    [
                        torch.stack(rank_flux)
                        for rank_flux in flux_means_gathered
                    ]
                )
            ]
            flux_squares = [
                torch.cat(
                    [
                        torch.stack(rank_flux)
                        for rank_flux in flux_squares_gathered
                    ]
                )
            ]
    else:
        means = [torch.cat(means, dim=0)]  # (B, d_features,)
        squares = [torch.cat(squares, dim=0)]  # (B, d_features,)
        flux_means = [torch.tensor(flux_means)]  # (B,)
        flux_squares = [torch.tensor(flux_squares)]  # (B,)

    if rank == 0:
        save_stats(
            static_dir_path,
            means,
            squares,
            flux_means,
            flux_squares,
            "parameter",
        )

    if distributed:
        dist.barrier()

    if rank == 0:
        print("Computing mean and std.-dev. for one-step differences...")
    ds_standard = WeatherDataset(
        datastore=datastore,
        split="train",
        ar_steps=ar_steps,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
    )
    if distributed:
        ds_standard = PaddedWeatherDataset(
            ds_standard,
            world_size,
            batch_size,
        )
        sampler_standard = DistributedSampler(
            ds_standard, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        sampler_standard = None
    loader_standard = torch.utils.data.DataLoader(
        ds_standard,
        batch_size,
        shuffle=False,
        num_workers=n_workers,
        sampler=sampler_standard,
    )

    # WeatherDataset no longer standardizes, so load the state mean/std saved
    # in the first pass above and apply them inline to compute diff stats.
    state_mean = torch.load(
        os.path.join(static_dir_path, "parameter_mean.pt"), weights_only=True
    )
    state_std = torch.load(
        os.path.join(static_dir_path, "parameter_std.pt"), weights_only=True
    )
    if distributed:
        state_mean = state_mean.to(device)
        state_std = state_std.to(device)

    time_step_int, time_step_unit = get_integer_time(step_length)
    assert (
        time_step_unit == "hours"
    ), "Only 'hours' time unit is supported by meps datastore."
    used_subsample_len = (65 // time_step_int) * time_step_int

    diff_means, diff_squares = [], []

    for init_batch, target_batch, _, _ in tqdm(
        loader_standard, disable=rank != 0
    ):
        if distributed:
            init_batch, target_batch = init_batch.to(device), target_batch.to(
                device
            )
        init_batch = (init_batch - state_mean) / state_std
        target_batch = (target_batch - state_mean) / state_std
        # (B, N_t', num_grid_nodes, num_state_vars)
        batch = torch.cat((init_batch, target_batch), dim=1)
        # Note: batch contains only 1h-steps
        stepped_batch = torch.cat(
            [
                batch[:, ss_i:used_subsample_len:time_step_int]
                for ss_i in range(time_step_int)
            ],
            dim=0,
        )
        # (B', N_t, num_grid_nodes, d_features),
        # B' = step_length*B
        batch_diffs = stepped_batch[:, 1:] - stepped_batch[:, :-1]
        # (B', N_t-1, num_grid_nodes, d_features)
        diff_means.append(torch.mean(batch_diffs, dim=(1, 2)).cpu())
        # (B', d_features,)
        diff_squares.append(torch.mean(batch_diffs**2, dim=(1, 2)).cpu())
        # (B', d_features,)

    if distributed and world_size > 1:
        dist.barrier()
        diff_means_gathered, diff_squares_gathered = (
            [None] * world_size,
            [None] * world_size,
        )
        dist.all_gather_object(
            diff_means_gathered, torch.cat(diff_means, dim=0)
        )
        dist.all_gather_object(
            diff_squares_gathered, torch.cat(diff_squares, dim=0)
        )

        if rank == 0:
            diff_means_gathered = torch.cat(diff_means_gathered, dim=0)
            diff_squares_gathered = torch.cat(diff_squares_gathered, dim=0)
            n_original_windows = (
                len(ds_standard.get_original_indices()) * time_step_int
            )
            diff_means = [diff_means_gathered[:n_original_windows]]
            diff_squares = [diff_squares_gathered[:n_original_windows]]

    diff_means = [torch.cat(diff_means, dim=0)]  # (B', d_features,)
    diff_squares = [torch.cat(diff_squares, dim=0)]  # (B', d_features,)

    if rank == 0:
        save_stats(static_dir_path, diff_means, diff_squares, [], [], "diff")

    if distributed:
        dist.destroy_process_group()


def cli():
    """Parse CLI arguments and trigger :func:`main`."""
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--datastore_config_path",
        type=str,
        help="Path to data config file",
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
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Run the script in distributed mode (default: False)",
    )
    args = parser.parse_args()
    distributed = bool(args.distributed)

    main(
        datastore_config_path=args.datastore_config_path,
        batch_size=args.batch_size,
        step_length=timedelta(hours=args.step_length),
        n_workers=args.n_workers,
        distributed=distributed,
    )


if __name__ == "__main__":
    cli()
