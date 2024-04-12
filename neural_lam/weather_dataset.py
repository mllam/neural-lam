# Standard library
import glob
import os
from datetime import datetime, timedelta

# Third-party
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr

# First-party
# BUG: Import should work in interactive mode as well -> create pypi package
from neural_lam import constants, utils

# pylint: disable=W0613:unused-argument
# pylint: disable=W0201:attribute-defined-outside-init


class WeatherDataset(torch.utils.data.Dataset):
    """
    N_t = 1h
    N_x = 582
    N_y = 390
    N_grid = 582*390 = 226980
    d_features = 4(features) * 21(vertical model levels) = 84
    d_forcing = 0
    #TODO: extract incoming radiation from KENDA
    """

    def __init__(
        self,
        dataset_name,
        path_verif_file=None,
        split="train",
        standardize=True,
        subset=False,
        batch_size=4,
        control_only=False,
    ):
        super().__init__()

        assert split in (
            "train",
            "val",
            "test",
            "pred",
            "verif",
        ), "Unknown dataset split"
        self.sample_dir_path = os.path.join(
            "data", dataset_name, "samples", split
        )

        self.batch_size = batch_size
        self.batch_index = 0
        self.index_within_batch = 0
        self.sample_dir_path = path_verif_file

        if split == "verif" and os.path.exists(self.sample_dir_path):
            self.np_files = np.load(self.sample_dir_path)
            self.split = split
            return

        self.zarr_files = sorted(
            glob.glob(os.path.join(self.sample_dir_path, "data*.zarr"))
        )
        if len(self.zarr_files) == 0:
            raise ValueError("No .zarr files found in directory")

        if subset:
            if constants.EVAL_DATETIME is not None and split == "test":
                eval_datetime_obj = datetime.strptime(
                    constants.EVAL_DATETIME, "%Y%m%d%H"
                )
                for i, file in enumerate(self.zarr_files):
                    file_datetime_str = file.split("/")[-1].split("_")[1][:-5]
                    file_datetime_obj = datetime.strptime(
                        file_datetime_str, "%Y%m%d%H"
                    )
                    if (
                        file_datetime_obj
                        <= eval_datetime_obj
                        < file_datetime_obj
                        + timedelta(hours=constants.CHUNK_SIZE)
                    ):
                        # Retrieve the current file and the next file if it
                        # exists
                        next_file_index = i + 1
                        if next_file_index < len(self.zarr_files):
                            self.zarr_files = [
                                file,
                                self.zarr_files[next_file_index],
                            ]
                        else:
                            self.zarr_files = [file]
                        position_within_file = int(
                            (
                                eval_datetime_obj - file_datetime_obj
                            ).total_seconds()
                            // 3600
                        )
                        self.batch_index = (
                            position_within_file // self.batch_size
                        )
                        self.index_within_batch = (
                            position_within_file % self.batch_size
                        )
                        break
            else:
                self.zarr_files = self.zarr_files[0:2]

            start_datetime = (
                self.zarr_files[0]
                .split("/")[-1]
                .split("_")[1]
                .replace(".zarr", "")
            )

            print("Data subset of 200 samples starts on the", start_datetime)

        # Separate 3D and 2D variables
        variables_3d = [
            var for var in constants.PARAM_NAMES_SHORT if constants.IS_3D[var]
        ]
        variables_2d = [
            var
            for var in constants.PARAM_NAMES_SHORT
            if not constants.IS_3D[var]
        ]

        # Stack 3D variables
        datasets_3d = [
            xr.open_zarr(file, consolidated=True)[variables_3d]
            .sel(z_1=constants.VERTICAL_LEVELS)
            .to_array()
            .stack(var=("variable", "z_1"))
            .transpose("time", "x_1", "y_1", "var")
            for file in self.zarr_files
        ]

        # Stack 2D variables without selecting along z_1
        datasets_2d = [
            xr.open_zarr(file, consolidated=True)[variables_2d]
            .to_array()
            .pipe(
                lambda ds: (ds if "z_1" in ds.dims else ds.expand_dims(z_1=[0]))
            )
            .stack(var=("variable", "z_1"))
            .transpose("time", "x_1", "y_1", "var")
            for file in self.zarr_files
        ]

        # Combine 3D and 2D datasets
        self.zarr_datasets = [
            xr.concat([ds_3d, ds_2d], dim="var").sortby("var")
            for ds_3d, ds_2d in zip(datasets_3d, datasets_2d)
        ]

        self.standardize = standardize
        if standardize:
            ds_stats = utils.load_dataset_stats(dataset_name, "cpu")
            if constants.GRID_FORCING_DIM > 0:
                (
                    self.data_mean,
                    self.data_std,
                    self.flux_mean,
                    self.flux_std,
                ) = (
                    ds_stats["data_mean"],
                    ds_stats["data_std"],
                    ds_stats["flux_mean"],
                    ds_stats["flux_std"],
                )
            else:
                self.data_mean, self.data_std = (
                    ds_stats["data_mean"],
                    ds_stats["data_std"],
                )
        self.random_subsample = split == "train"
        self.split = split

    def __len__(self):
        num_steps = (
            constants.TRAIN_HORIZON
            if self.split == "train"
            else constants.EVAL_HORIZON
        )
        total_time = 1
        if hasattr(self, "zarr_files"):
            total_time = len(self.zarr_files) * constants.CHUNK_SIZE - num_steps
        return total_time

    def __getitem__(self, idx):
        if self.split == "verif":
            return self.np_files

        num_steps = (
            constants.TRAIN_HORIZON
            if self.split == "train"
            else constants.EVAL_HORIZON
        )

        # Calculate which zarr files need to be loaded
        start_file_idx = idx // constants.CHUNK_SIZE
        end_file_idx = (idx + num_steps) // constants.CHUNK_SIZE
        # Index of current slice
        idx_sample = idx % constants.CHUNK_SIZE

        sample_archive = xr.concat(
            self.zarr_datasets[start_file_idx : end_file_idx + 1],
            dim="time",
        )

        sample_xr = sample_archive.isel(
            time=slice(idx_sample, idx_sample + num_steps)
        )

        # (N_t', N_x, N_y, d_features')
        sample = torch.tensor(sample_xr.values, dtype=torch.float32)

        sample = sample.flatten(1, 2)  # (N_t, N_grid, d_features)

        if self.standardize:
            # Standardize sample
            sample = (sample - self.data_mean) / self.data_std

        # Split up sample in init. states and target states
        init_states = sample[:2]  # (2, N_grid, d_features)
        target_states = sample[2:]  # (sample_length-2, N_grid, d_features)

        return init_states, target_states


class WeatherDataModule(pl.LightningDataModule):
    """DataModule for weather data."""

    def __init__(
        self,
        dataset_name,
        split="train",
        path_verif_file=None,
        standardize=True,
        subset=False,
        batch_size=4,
        num_workers=16,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.path_verif_file = path_verif_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.standardize = standardize
        self.subset = subset

    def prepare_data(self):
        # download, split, etc... called only on 1 GPU/TPU in distributed
        pass

    def setup(self, stage=None):
        # make assignments here (val/train/test/pred split)
        # called on every process in DDP
        if stage == "fit" or stage is None:
            self.train_dataset = WeatherDataset(
                self.dataset_name,
                split="train",
                standardize=self.standardize,
                subset=self.subset,
                batch_size=self.batch_size,
            )
            self.val_dataset = WeatherDataset(
                self.dataset_name,
                split="val",
                standardize=self.standardize,
                subset=self.subset,
                batch_size=self.batch_size,
            )

        if stage == "test" or stage is None:
            self.test_dataset = WeatherDataset(
                self.dataset_name,
                split="test",
                standardize=self.standardize,
                subset=self.subset,
                batch_size=self.batch_size,
            )

        if stage == "verif":
            self.verif_dataset = WeatherDataset(
                self.dataset_name,
                self.path_verif_file,
                split="verif",
                standardize=False,
                subset=False,
                batch_size=self.batch_size,
            )

        if stage == "pred" or stage is None:
            self.pred_dataset = WeatherDataset(
                self.dataset_name,
                split="pred",
                standardize=self.standardize,
                subset=False,
                batch_size=1,
            )

    def train_dataloader(self):
        """Load train dataset."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def val_dataloader(self):
        """Load validation dataset."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size // self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self):
        """Load test dataset."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def pred_dataloader(self):
        """Load prediction dataset."""
        return torch.utils.data.DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
        )

    def verif_dataloader(self):
        """Load inference output dataset."""
        return torch.utils.data.DataLoader(
            self.verif_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
        )
