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
        self.config_loader = config.Config(data_config)

        self.state = self.config_loader.process_dataset("state", self.split)
        assert self.state is not None, "State dataset not found"
        self.forcing = self.config_loader.process_dataset(
            "forcing", self.split
        )
        self.boundary = self.config_loader.process_dataset(
            "boundary", self.split
        )

        self.state_times = self.state.time.values
        self.forcing_window = self.config_loader.forcing.window
        self.boundary_window = self.config_loader.boundary.window

        if self.forcing is not None:
            self.forcing_windowed = (
                self.forcing.sel(
                    time=self.state.time,
                    method="nearest",
                )
                .pad(
                    time=(self.forcing_window // 2, self.forcing_window // 2),
                    mode="edge",
                )
                .rolling(time=self.forcing_window, center=True)
                .construct("window")
            )

        if self.boundary is not None:
            self.boundary_windowed = (
                self.boundary.sel(
                    time=self.state.time,
                    method="nearest",
                )
                .pad(
                    time=(
                        self.boundary_window // 2,
                        self.boundary_window // 2,
                    ),
                    mode="edge",
                )
                .rolling(time=self.boundary_window, center=True)
                .construct("window")
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
            self.forcing_windowed.isel(
                time=slice(idx + 2, idx + self.ar_steps)
            )
            .stack(variable_window=("variable", "window"))
            .values
            if self.forcing is not None
            else torch.tensor([])
        )

        boundary = (
            self.boundary_windowed.isel(
                time=slice(idx + 2, idx + self.ar_steps)
            )
            .stack(variable_window=("variable", "window"))
            .values
            if self.boundary is not None
            else torch.tensor([])
        )

        init_states = sample[:2]
        target_states = sample[2:]

        batch_times = (
            self.state.isel(time=slice(idx + 2, idx + self.ar_steps))
            .time.values.astype(str)
            .tolist()
        )

        # init_states: (2, N_grid, d_features)
        # target_states: (ar_steps-2, N_grid, d_features)
        # forcing: (ar_steps-2, N_grid, d_windowed_forcing)
        # boundary: (ar_steps-2, N_grid, d_windowed_boundary)
        # batch_times: (ar_steps-2,)
        return init_states, target_states, forcing, boundary, batch_times


class WeatherDataModule(pl.LightningDataModule):
    """DataModule for weather data."""

    def __init__(
        self,
        ar_steps_train=3,
        ar_steps_eval=25,
        batch_size=4,
        num_workers=16,
    ):
        super().__init__()
        self.ar_steps_train = ar_steps_train
        self.ar_steps_eval = ar_steps_eval
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
                batch_size=self.batch_size,
            )
            self.val_dataset = WeatherDataset(
                split="val",
                ar_steps=self.ar_steps_eval,
                batch_size=self.batch_size,
            )

        if stage == "test" or stage is None:
            self.test_dataset = WeatherDataset(
                split="test",
                ar_steps=self.ar_steps_eval,
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
