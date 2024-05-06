# Standard library
import os
from functools import lru_cache

# Third-party
import pytorch_lightning as pl
import torch
import xarray as xr
import yaml


class ConfigLoader:
    """
    Class for loading configuration files.

    This class loads a YAML configuration file and provides a way to access
    its values as attributes.
    """

    def __init__(self, config_path, values=None):
        self.config_path = config_path
        if values is None:
            self.values = self.load_config()
        else:
            self.values = values

    def load_config(self):
        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)

    @lru_cache(maxsize=None)
    def __getattr__(self, name):
        keys = name.split(".")
        value = self.values
        for key in keys:
            if key in value:
                value = value[key]
            else:
                None
        if isinstance(value, dict):
            return ConfigLoader(None, values=value)
        return value

    def __getitem__(self, key):
        value = self.values[key]
        if isinstance(value, dict):
            return ConfigLoader(None, values=value)
        return value

    def __contains__(self, key):
        return key in self.values


class WeatherDataset(torch.utils.data.Dataset):
    """
    Dataset class for weather data.

    This class loads and processes weather data from zarr files based on the
    provided configuration. It supports splitting the data into train,
    validation, and test sets.
    """

    def process_dataset(self, dataset_name):
        """
        Process a single dataset specified by the dataset name.

        Args:
            dataset_name (str): Name of the dataset to process.

        Returns:
            xarray.Dataset: Processed dataset.
        """

        dataset_path = self.config_loader.zarrs[dataset_name].path
        if dataset_path is None or not os.path.exists(dataset_path):
            print(f"Dataset '{dataset_name}' not found at path: {dataset_path}")
            return None
        dataset = xr.open_zarr(dataset_path, consolidated=True)

        start, end = (
            self.config_loader.splits[self.split].start,
            self.config_loader.splits[self.split].end,
        )
        dataset = dataset.sel(time=slice(start, end))
        dataset = dataset.rename_dims(
            {
                v: k
                for k, v in self.config_loader.zarrs[dataset_name].dims.values.items()
                if k not in dataset.dims
            }
        )
        if "grid" not in dataset.dims:
            dataset = dataset.stack(grid=("x", "y"))

        vars_surface = []
        if self.config_loader[dataset_name].surface:
            vars_surface = dataset[self.config_loader[dataset_name].surface]

        vars_atmosphere = []
        if self.config_loader[dataset_name].atmosphere:
            vars_atmosphere = xr.merge(
                [
                    dataset[var].sel(level=level, drop=True).rename(f"{var}_{level}")
                    for var in self.config_loader[dataset_name].atmosphere
                    for level in self.config_loader[dataset_name].levels
                ]
            )

        if vars_surface and vars_atmosphere:
            dataset = xr.merge([vars_surface, vars_atmosphere])
        elif vars_surface:
            dataset = vars_surface
        elif vars_atmosphere:
            dataset = vars_atmosphere
        else:
            print("No variables found in dataset {dataset_name}")
            return None

        dataset = dataset.squeeze(drop=True).to_array()
        if "time" in dataset.dims:
            dataset = dataset.transpose("time", "grid", "variable")
        else:
            dataset = dataset.transpose("grid", "variable")
        return dataset

    def __init__(
        self,
        split="train",
        batch_size=4,
        ar_steps=3,
        control_only=False,
        yaml_path="neural_lam/data_config.yaml",
    ):
        super().__init__()

        assert split in (
            "train",
            "val",
            "test",
        ), "Unknown dataset split"

        self.split = split
        self.batch_size = batch_size
        self.ar_steps = ar_steps
        self.control_only = control_only
        self.config_loader = ConfigLoader(yaml_path)

        self.state = self.process_dataset("state")
        assert self.state is not None, "State dataset not found"
        self.static = self.process_dataset("static")
        self.forcings = self.process_dataset("forcing")
        self.boundary = self.process_dataset("boundary")

        if self.static is not None:
            self.static = self.static.expand_dims({"time": self.state.time}, axis=0)
            self.state = xr.concat([self.state, self.static], dim="variable")

    def __len__(self):
        return len(self.state.time) - self.ar_steps

    def __getitem__(self, idx):
        sample = torch.tensor(
            self.state.isel(time=slice(idx, idx + self.ar_steps)).values,
            dtype=torch.float32,
        )

        forcings = torch.tensor(
            self.forcings.isel(time=slice(idx, idx + self.ar_steps)).values,
            dtype=torch.float32,
        ) if self.forcings is not None else torch.tensor([])

        boundary = torch.tensor(
            self.boundary.isel(time=slice(idx, idx + self.ar_steps)).values,
            dtype=torch.float32,
        ) if self.boundary is not None else torch.tensor([])

        init_states = sample[:2]
        target_states = sample[2:]

        batch_times = (
            self.state.isel(time=slice(idx, idx + self.ar_steps))
            .time.values.astype(str)
            .tolist()
        )

        # init_states: (2, N_grid, d_features)
        # target_states: (ar_steps-2, N_grid, d_features)
        # forcings: (ar_steps, N_grid, d_windowed_forcings)
        # boundary: (ar_steps, N_grid, d_windowed_boundary)
        # batch_times: (ar_steps,)
        return init_states, target_states, forcings, boundary, batch_times


class WeatherDataModule(pl.LightningDataModule):
    """DataModule for weather data."""

    def __init__(
        self,
        batch_size=4,
        num_workers=16,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = WeatherDataset(
                split="train",
                batch_size=self.batch_size,
            )
            self.val_dataset = WeatherDataset(
                split="val",
                batch_size=self.batch_size,
            )

        if stage == "test" or stage is None:
            self.test_dataset = WeatherDataset(
                split="test",
                batch_size=self.batch_size,
            )

    def train_dataloader(self):
        """Load train dataset."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def val_dataloader(self):
        """Load validation dataset."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        """Load test dataset."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


data_module = WeatherDataModule(batch_size=4, num_workers=0)
data_module.setup()
train_dataloader = data_module.train_dataloader()
for batch in train_dataloader:
    print(batch[0].shape)
    print(batch[1].shape)
    print(batch[2].shape)
    print(batch[3].shape)
    print(batch[4])
    break
