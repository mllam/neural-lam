import glob
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
    d_features = 4(features) * 7(vertical model levels) = 28
    d_forcing = 0 #TODO: extract incoming radiation from KENDA
    """

    def __init__(self, dataset_name, split="train", standardize=True, subset=False):
        super().__init__()

        assert split in ("train", "val", "test"), "Unknown dataset split"
        sample_dir_path = os.path.join("data", dataset_name, "samples", split)

        self.zarr_files = sorted(glob.glob(
            os.path.join(sample_dir_path, "data*.zarr")))
        if len(self.zarr_files) == 0:
            raise ValueError("No .zarr files found in directory")

        if subset:
            # Limit to 200 samples
            self.zarr_files = self.zarr_files[constants.
                                              eval_sample: constants.eval_sample + 2]
            start_date = self.zarr_files[0].split(
                "/")[-1].split("_")[1].replace('.zarr', '')

            print("Evaluation on subset of 200 samples")
            print("Evaluation starts on the", start_date)

        self.zarr_datasets = [
            xr.open_zarr(
                file,
                consolidated=True)[
                constants.param_names_short].sel(
                z_1=constants.vertical_levels).to_array().stack(
                    var=(
                        'variable',
                        'z_1')).transpose(
                            "time",
                            "x_1",
                            "y_1",
                "var") for file in self.zarr_files]

        self.standardize = standardize
        if standardize:
            ds_stats = utils.load_dataset_stats(dataset_name, "cpu")
            self.data_mean, self.data_std = ds_stats["data_mean"], ds_stats["data_std"]

        self.random_subsample = split == "train"
        self.split = split

    def __len__(self):
        num_steps = constants.train_horizon if self.split == "train" else constants.eval_horizon
        total_time = len(
            self.zarr_files) * constants.data_config["chunk_size"] - num_steps
        return total_time

    def __getitem__(self, idx):
        num_steps = constants.train_horizon if self.split == "train" else constants.eval_horizon

        # Calculate which zarr files need to be loaded
        start_file_idx = idx // constants.data_config["chunk_size"]
        end_file_idx = (idx + num_steps) // constants.data_config["chunk_size"]
        # Index of current slice
        idx_sample = idx % constants.data_config["chunk_size"]

        sample_archive = xr.concat(
            self.zarr_datasets[start_file_idx: end_file_idx + 1],
            dim='time')

        sample_xr = sample_archive.isel(time=slice(idx_sample, idx_sample + num_steps))

        # (N_t', N_x, N_y, d_features')
        sample = torch.tensor(sample_xr.values, dtype=torch.float32)

        sample = sample.flatten(1, 2)  # (N_t, N_grid, d_features)

        if self.standardize:
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
            self.val_dataset, batch_size=self.batch_size // self.batch_size,
            num_workers=self.num_workers // 2, shuffle=False, pin_memory=False,)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size // self.batch_size,
            num_workers=self.num_workers // 2, shuffle=False, pin_memory=False)
