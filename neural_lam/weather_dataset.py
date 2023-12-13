import os

import pytorch_lightning as pl
import torch
import xarray as xr

# BUG: Import should work in interactive mode as well -> create pypi package
from neural_lam import constants, utils


class WeatherDataset(torch.utils.data.Dataset):
    """
    For our dataset:
    N_t = 1h
    N_x = 582
    N_y = 390
    N_grid = 582*390 = 226980
    d_features = 8(features) * 7(vertical model levels) = 56
    d_forcing = 0 #TODO: extract incoming radiation from KENDA
    """

    def __init__(self, dataset_name, split="train", standardize=True, subset=False):
        super().__init__()

        assert split in ("train", "val", "test"), "Unknown dataset split"
        self.sample_dir_path = os.path.join("data", dataset_name, "samples", split)

        zarr_files = os.path.join(self.sample_dir_path, "data_" + split + ".zarr")
        if not zarr_files:
            raise ValueError("No .zarr files found in directory")
        self.sample_archive = xr.open_zarr(zarr_files, consolidated=True)

        if subset:
            # Limit to 200 samples
            self.sample_archive = self.sample_archive.isel(
                time=slice(constants.eval_sample, constants.eval_sample + 200))

        # Set up for standardization
        self.standardize = standardize
        if standardize:
            ds_stats = utils.load_dataset_stats(dataset_name, "cpu")
            self.data_mean, self.data_std = ds_stats["data_mean"], ds_stats["data_std"]

        # If subsample index should be sampled (only duing training)
        self.random_subsample = split == "train"
        self.split = split

    def __len__(self):
        num_steps = constants.train_horizon if self.split == "train" else constants.eval_horizon
        total_time = self.sample_archive.time.size - num_steps + 1
        return total_time

    def __getitem__(self, idx):
        num_steps = constants.train_horizon if self.split == "train" else constants.eval_horizon
        total_time = self.sample_archive.time.size - num_steps + 1
        time_idx = idx % total_time
        sample = self.sample_archive.isel(time=slice(time_idx, time_idx + num_steps))
        sample = sample[constants.param_names_short]
        for var in sample.data_vars:
            for level in constants.vertical_levels:
                new_var_name = f"{var}_z{level}"
                sample[new_var_name] = sample[var].sel(z_1=level)

        da = sample.drop_dims("z_1").to_array().transpose(
            "time", "x_1", "y_1", "variable").values

        sample = torch.tensor(da, dtype=torch.float32)
        # (N_t', N_x, N_y, d_features')

        # Flatten spatial dim
        sample = sample.flatten(1, 2)  # (N_t, N_grid, d_features)

        if self.standardize:
            # Standardize sample
            sample = (sample - self.data_mean) / self.data_std

        # Split up sample in init. states and target states
        init_states = sample[:2]  # (2, N_grid, d_features)
        target_states = sample[2:]  # (sample_length-2, N_grid, d_features)

        return init_states, target_states


class WeatherDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, split="train", standardize=True,
                 subset=False, batch_size=4, num_workers=16):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.standardize = standardize
        self.subset = subset

    def prepare_data(self):
        # download, split, etc...
        # called only on 1 GPU/TPU in distributed
        pass

    def setup(self, stage=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        if stage == 'fit' or stage is None:
            self.train_dataset = WeatherDataset(
                self.dataset_name,
                split="train",
                standardize=self.standardize,
                subset=self.subset)
            self.val_dataset = WeatherDataset(
                self.dataset_name,
                split="val",
                standardize=self.standardize,
                subset=self.subset)

        if stage == 'test' or stage is None:
            self.test_dataset = WeatherDataset(
                self.dataset_name,
                split="test",
                standardize=self.standardize,
                subset=self.subset)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, pin_memory=False,)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, pin_memory=False,
            # persistent_workers=True,
            # drop_last=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, pin_memory=False)
