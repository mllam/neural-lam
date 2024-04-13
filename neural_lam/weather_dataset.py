# Standard library
import os
from datetime import datetime, timedelta
from random import randint

# Third-party
import numpy as np
import pytorch_lightning as pl
import torch
import xarray as xr

# First-party
from neural_lam import constants, utils


class WeatherDataset(torch.utils.data.Dataset):
    """Weather dataset for PyTorch Lightning."""

    # pylint: disable=too-many-branches
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
            "predict",
            "verif",
        ), "Unknown dataset split"

        if split == "verif":
            self.np_files = np.load(path_verif_file)
            self.split = split
            return

        self.zarr_path = os.path.join(
            "data", dataset_name, "samples", split, "data.zarr"
        )
        self.ds = xr.open_zarr(self.zarr_path, consolidated=True)
        if split == "train":
            self.ds = self.ds.sel(time=slice("2015", "2019"))
        else:
            # BUG: Clean this up after zarr archive is fixed
            self.ds = self.ds.sel(time=slice("2015", "2020"))

        new_vars = {}
        for var_name, data_array in self.ds.data_vars.items():
            if var_name in constants.PARAM_NAMES_SHORT:
                if constants.IS_3D[var_name]:
                    for z in constants.VERTICAL_LEVELS:
                        new_key = f"{var_name}_{int(z)}"
                        new_vars[new_key] = data_array.sel(z=z).drop_vars("z")
                # BUG: Clean this up after zarr archive is fixed
                elif var_name == "T_2M":
                    new_vars[var_name] = data_array.sel(z=2).drop_vars("z")
                elif var_name in ["U_10M", "V_10M"]:
                    new_vars[var_name] = data_array.sel(z=10).drop_vars("z")
                elif var_name == "PMSL":
                    new_vars[var_name] = data_array.sel(z=0).drop_vars("z")
                else:
                    new_vars[var_name] = data_array

        self.ds = (
            xr.Dataset(new_vars)
            # BUG: This should not be necessary with clean data without nans
            .drop_isel(time=848)
            .to_array()
            .transpose("time", "x", "y", "variable")
        )

        if subset:
            if constants.EVAL_DATETIMES is not None and split == "test":
                eval_datetime_obj = datetime.strptime(
                    constants.EVAL_DATETIMES[0], "%Y%m%d%H"
                )
                self.ds = self.ds.sel(
                    time=slice(
                        eval_datetime_obj,
                        eval_datetime_obj + timedelta(hours=50),
                    )
                )
            else:
                start_idx = randint(0, self.ds.time.size - 50)
                self.ds = self.ds.isel(time=slice(start_idx, start_idx + 50))

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
        self.num_steps = (
            constants.TRAIN_HORIZON
            if self.split == "train"
            else constants.EVAL_HORIZON
        )
        self.batch_size = batch_size
        self.control_only = control_only

    def __len__(self):
        if self.split == "verif":
            return len(self.np_files)
        return len(self.ds.time) - self.num_steps

    def __getitem__(self, idx):
        if self.split == "verif":
            return self.np_files
        sample_xr = self.ds.isel(time=slice(idx, idx + self.num_steps))

        # (N_t', N_x, N_y, d_features')
        sample = torch.tensor(sample_xr.values, dtype=torch.float32)
        sample = sample.flatten(1, 2)  # (N_t, N_grid, d_features)

        if self.standardize:
            sample = (sample - self.data_mean) / self.data_std

        init_states = sample[:2]  # (2, N_grid, d_features)
        target_states = sample[2:]  # (sample_length-2, N_grid, d_features)

        batch_time = self.ds.isel(time=idx).time.values
        batch_time = np.datetime_as_string(batch_time, unit="h")
        batch_time = str(batch_time).replace("-", "").replace("T", "")
        return init_states, target_states, batch_time


class WeatherDataModule(pl.LightningDataModule):
    """DataModule for weather data."""

    def __init__(
        self,
        dataset_name,
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
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.verif_dataset = None
        self.predict_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = WeatherDataset(
                dataset_name=self.dataset_name,
                split="train",
                standardize=self.standardize,
                subset=self.subset,
                batch_size=self.batch_size,
            )
            self.val_dataset = WeatherDataset(
                dataset_name=self.dataset_name,
                split="val",
                standardize=self.standardize,
                subset=self.subset,
                batch_size=self.batch_size,
            )

        if stage == "test" or stage is None:
            self.test_dataset = WeatherDataset(
                dataset_name=self.dataset_name,
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

        if stage == "predict" or stage is None:
            self.predict_dataset = WeatherDataset(
                self.dataset_name,
                split="predict",
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
            batch_size=self.batch_size,
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

    def predict_dataloader(self):
        """Load prediction dataset."""
        return torch.utils.data.DataLoader(
            self.predict_dataset,
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
