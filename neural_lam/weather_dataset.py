# Third-party
import pytorch_lightning as pl
import torch

# First-party
from neural_lam import utils


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
        batch_size=4,
        ar_steps=3,
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
        self.config_loader = utils.ConfigLoader(data_config)

        self.state = self.config_loader.process_dataset("state", self.split)
        assert self.state is not None, "State dataset not found"
        self.forcings = self.config_loader.process_dataset(
            "forcing", self.split
        )
        self.boundary = self.config_loader.process_dataset(
            "boundary", self.split
        )

        self.state_times = self.state.time.values
        self.forcing_window = self.config_loader.forcing.window
        self.boundary_window = self.config_loader.boundary.window
        self.idx_max = max(
            (self.boundary_window - 1), (self.forcing_window - 1)
        )

        if self.forcings is not None:
            self.forcings_windowed = (
                self.forcings.sel(
                    time=self.forcings.time.isin(self.state.time),
                    method="nearest",
                )
                .rolling(time=self.forcing_window, center=True)
                .construct("window")
            )
        if self.boundary is not None:
            self.boundary_windowed = (
                self.boundary.sel(
                    time=self.forcings.time.isin(self.state.time),
                    method="nearest",
                )
                .rolling(time=self.boundary_window, center=True)
                .construct("window")
            )

    def __len__(self):
        # Skip first and last time step
        return len(self.state.time) - self.ar_steps - self.idx_max

    def __getitem__(self, idx):
        idx += self.idx_max / 2  # Skip first time step
        sample = torch.tensor(
            self.state.isel(time=slice(idx, idx + self.ar_steps)).values,
            dtype=torch.float32,
        )

        forcings = (
            self.forcings_windowed.isel(time=slice(idx, idx + self.ar_steps))
            .stack(variable_window=("variable", "window"))
            .values
            if self.forcings is not None
            else torch.tensor([])
        )

        boundary = (
            self.boundary_windowed.isel(time=slice(idx, idx + self.ar_steps))
            .stack(variable_window=("variable", "window"))
            .values
            if self.boundary is not None
            else torch.tensor([])
        )

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
    print(batch[2][0, 0, 0, :])
    break
