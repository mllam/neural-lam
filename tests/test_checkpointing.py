# Standard library
from pathlib import Path
from unittest.mock import MagicMock, patch

# Third-party
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

# First-party
from neural_lam.train_model import main


@patch("neural_lam.train_model.MODELS", {"graph_lam": MagicMock()})
def test_checkpoint_callbacks_configured():
    """Train setup should include both validation and rescue checkpoints."""
    test_args = [
        "train_model.py",
        "--config_path",
        "dummy.yaml",
        "--epochs",
        "1",
    ]

    with (
        patch("neural_lam.train_model.load_config_and_datastore") as mock_load,
        patch("neural_lam.train_model.utils.setup_training_logger"),
        patch("neural_lam.train_model.pl.Trainer") as mock_trainer,
    ):
        mock_load.return_value = (MagicMock(), MagicMock())
        main(input_args=test_args[1:])

        _, kwargs = mock_trainer.call_args
        callbacks = kwargs.get("callbacks", [])

        assert len(callbacks) == 2
        filenames = [cb.filename for cb in callbacks]
        assert "min_val_loss" in filenames
        assert "last" in filenames


class TinyModule(pl.LightningModule):
    """Small module to test checkpoint callback behavior quickly in CI."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_last_checkpoint_saved_without_validation(tmp_path):
    """
    Rescue checkpoint should be written at train epoch end even without validation.
    """
    train_data = TensorDataset(torch.randn(8, 1), torch.randn(8, 1))
    train_loader = DataLoader(train_data, batch_size=2)

    ckpt_dir = Path(tmp_path) / "checkpoints"
    val_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="min_val_loss",
        monitor="val_mean_loss",
        mode="min",
        save_top_k=1,
    )
    latest_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="last",
        monitor=None,
        save_top_k=1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        enable_version_counter=False,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_model_summary=False,
        callbacks=[val_checkpoint, latest_checkpoint],
        limit_val_batches=0,
    )
    trainer.fit(TinyModule(), train_dataloaders=train_loader)

    assert (ckpt_dir / "last.ckpt").exists()
