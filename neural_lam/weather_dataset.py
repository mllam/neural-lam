# Standard library
import datetime as dt
import os
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
        subset=0,
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

        self.split = split
        self.batch_size = batch_size
        self.control_only = control_only
        self.subset = subset

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
            self.ds = self.ds.sel(time=slice("2020-01-01", "2020-12-31"))

        new_vars = {}
        forcings = {}
        for var_name, data_array in self.ds.data_vars.items():
            if var_name in constants.PARAM_NAMES_SHORT:
                if constants.IS_3D[var_name]:
                    for z in constants.VERTICAL_LEVELS:
                        new_key = f"{var_name}_{int(z)}"
                        new_vars[new_key] = data_array.sel(z=z).drop_vars("z")
                else:
                    new_vars[var_name] = data_array
            elif var_name in constants.FORCING_NAMES_SHORT:
                forcings[var_name] = data_array

        self.ds = (
            xr.Dataset(new_vars)
            .to_array()
            .transpose("time", "x", "y", "variable")
        )
        self.forcings = (
            xr.Dataset(forcings)
            .to_array()
            .transpose("time", "x", "y", "variable")
        )

        self.num_steps = (
            constants.TRAIN_HORIZON
            if self.split == "train"
            else constants.EVAL_HORIZON
        )

        if subset > 0:
            if constants.EVAL_DATETIMES is not None and split == "test":
                utils.rank_zero_print(
                    f"Subsetting test dataset, using only first "
                    f"{self.num_steps} hours after "
                    f"{constants.EVAL_DATETIMES[0]}"
                )
                eval_datetime_obj = dt.datetime.strptime(
                    constants.EVAL_DATETIMES[0], "%Y%m%d%H"
                )
                init_datetime = np.datetime64(eval_datetime_obj, "ns")
                end_datetime = np.datetime64(
                    eval_datetime_obj + dt.timedelta(hours=self.num_steps), "ns"
                )
                assert (
                    init_datetime in self.ds.time.values
                ), f"Eval datetime {init_datetime} not in dataset. "
                self.ds = self.ds.sel(
                    time=slice(
                        init_datetime,
                        end_datetime,
                    )
                )
                self.forcings = self.forcings.sel(
                    time=slice(
                        init_datetime,
                        end_datetime,
                    )
                )
            else:
                start_idx = randint(0, self.ds.time.size - self.subset)
                self.ds = self.ds.isel(
                    time=slice(start_idx, start_idx + self.subset)
                )
                self.forcings = self.forcings.isel(
                    time=slice(start_idx, start_idx + self.subset)
                )

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

    def __len__(self):
        if self.split == "verif":
            return len(self.np_files)
        return len(self.ds.time) - self.num_steps

    def __getitem__(self, idx):
        if self.split == "verif":
            return self.np_files
        sample_xr = self.ds.isel(time=slice(idx, idx + self.num_steps))
        forcings = self.forcings.isel(time=slice(idx, idx + self.num_steps))

        # (N_t', N_x, N_y, d_features')
        sample = torch.tensor(sample_xr.values, dtype=torch.float32)
        forcings = torch.tensor(forcings.values, dtype=torch.float32)
        sample = sample.flatten(1, 2)  # (N_t, N_grid, d_features)
        forcings = forcings.flatten(1, 2)  # (N_t, N_grid, d_forcing)

        if self.standardize:
            sample = (sample - self.data_mean) / self.data_std
            forcings = (forcings - self.flux_mean) / self.flux_std

        init_states = sample[:2]  # (2, N_grid, d_features)
        target_states = sample[2:]  # (sample_length-2, N_grid, d_features)

        batch_times = self.ds.isel(
            time=slice(idx, idx + self.num_steps)
        ).time.values
        batch_times = np.datetime_as_string(batch_times, unit="h")
        batch_times = [
            str(t).replace("-", "").replace("T", "") for t in batch_times
        ]

        # Time of day and year
        dt_objs = [dt.datetime.strptime(t, "%Y%m%d%H") for t in batch_times]
        hours_of_day = [dt_obj.hour for dt_obj in dt_objs]
        seconds_into_year = [
            (dt_obj - dt.datetime(dt_obj.year, 1, 1)).total_seconds()
            for dt_obj in dt_objs
        ]

        hour_angles = torch.tensor(
            [(hour_of_day / 12) * torch.pi for hour_of_day in hours_of_day]
        )  # (sample_len,)
        year_angles = torch.tensor(
            [
                (second_into_year / constants.SECONDS_IN_YEAR) * 2 * torch.pi
                for second_into_year in seconds_into_year
            ]
        )  # (sample_len,)
        datetime_forcing = torch.stack(
            (
                torch.sin(hour_angles),
                torch.cos(hour_angles),
                torch.sin(year_angles),
                torch.cos(year_angles),
            ),
            dim=1,
        )  # (sample_len, 4)
        datetime_forcing = (datetime_forcing + 1) / 2  # Rescale to [0,1]

        datetime_forcing = datetime_forcing.unsqueeze(1).expand(
            -1, forcings.shape[1], -1
        )  # (sample_len, N_grid, 4)

        # Put forcing features together
        forcings = torch.cat(
            (forcings, datetime_forcing), dim=-1
        )  # (sample_len, N_grid, d_forcing)

        # Combine forcing over each window of 3 time steps (prev_prev, prev,
        # current)
        forcing = torch.cat(
            (
                forcings[:-2],
                forcings[1:-1],
                forcings[2:],
            ),
            dim=2,
        )  # (sample_len-2, N_grid, 3*d_forcing)
        # Now index 0 of ^ corresponds to forcing at index 0-2 of sample

        # Start the plotting at the first time step
        batch_time = batch_times[0]
        return init_states, target_states, batch_time, forcing


class WeatherDataModule(pl.LightningDataModule):
    """DataModule for weather data."""

    def __init__(
        self,
        dataset_name,
        path_verif_file=None,
        standardize=True,
        subset=0,
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
                subset=0,
                batch_size=self.batch_size,
            )

        if stage == "predict" or stage is None:
            self.predict_dataset = WeatherDataset(
                self.dataset_name,
                split="predict",
                standardize=self.standardize,
                subset=0,
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
