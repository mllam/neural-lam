# Third-party
import pytorch_lightning as pl
import torch

# First-party
from neural_lam import config


class WeatherDataset(torch.utils.data.Dataset):
    """
    Dataset class for weather data.

    This class loads and processes weather data from zarr files based on the
    provided configuration. It supports splitting the data into train,
    validation, and test sets.
    """

    def __init__(
        self,
        split="train",
        ar_steps=3,
        batch_size=4,
        standardize=True,
        control_only=False,
        data_config="neural_lam/data_config.yaml",
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
        self.config_loader = config.Config.from_file(data_config)

        self.state = self.config_loader.process_dataset("state", self.split)
        assert self.state is not None, "State dataset not found"
        self.forcing = self.config_loader.process_dataset(
            "forcing", self.split
        )
        self.state_times = self.state.time.values

        # Set up for standardization
        # NOTE: This will become part of ar_model.py soon!
        self.standardize = standardize
        if standardize:
            state_stats = self.config_loader.load_normalization_stats(
                "state", datatype="torch"
            )
            self.state_mean, self.state_std = (
                state_stats["state_mean"],
                state_stats["state_std"],
            )

            if self.forcing is not None:
                forcing_stats = self.config_loader.load_normalization_stats(
                    "forcing", datatype="torch"
                )
                self.forcing_mean, self.forcing_std = (
                    forcing_stats["forcing_mean"],
                    forcing_stats["forcing_std"],
                )

    def __len__(self):
        # Skip first and last time step
        return len(self.state.time) - self.ar_steps

    def __getitem__(self, idx):
        sample = torch.tensor(
            self.state.isel(time=slice(idx, idx + self.ar_steps)).values,
            dtype=torch.float32,
        )

        forcing = (
            torch.tensor(
                self.forcing.isel(
                    time=slice(idx + 2, idx + self.ar_steps)
                ).values
            )
            if self.forcing is not None
            else torch.tensor([])
        )

        init_states = sample[:2]
        target_states = sample[2:]

        batch_times = (
            self.state.isel(time=slice(idx + 2, idx + self.ar_steps))
            .time.values.astype(str)
            .tolist()
        )

        if self.standardize:
            init_states = (init_states - self.state_mean) / self.state_std
            target_states = (target_states - self.state_mean) / self.state_std

            if self.forcing is not None:
                forcing = (forcing - self.forcing_mean) / self.forcing_std

        # init_states: (2, N_grid, d_features)
        # target_states: (ar_steps-2, N_grid, d_features)
        # forcing: (ar_steps-2, N_grid, d_windowed_forcing)
        # batch_times: (ar_steps-2,)
        return init_states, target_states, forcing, batch_times


class WeatherDataModule(pl.LightningDataModule):
    """DataModule for weather data."""

    def __init__(
        self,
        ar_steps_train=3,
        ar_steps_eval=25,
        standardize=True,
        batch_size=4,
        num_workers=16,
    ):
        super().__init__()
        self.ar_steps_train = ar_steps_train
        self.ar_steps_eval = ar_steps_eval
        self.standardize = standardize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = WeatherDataset(
                split="train",
                ar_steps=self.ar_steps_train,
                standardize=self.standardize,
                batch_size=self.batch_size,
            )
            self.val_dataset = WeatherDataset(
                split="val",
                ar_steps=self.ar_steps_eval,
                standardize=self.standardize,
                batch_size=self.batch_size,
            )

        if stage == "test" or stage is None:
            self.test_dataset = WeatherDataset(
                split="test",
                ar_steps=self.ar_steps_eval,
                standardize=self.standardize,
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
