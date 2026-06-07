# Standard library
from pathlib import Path

# Third-party
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


class TinyModule(pl.LightningModule):
    """Small module to exercise checkpoint callbacks quickly in CI."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_last_checkpoint_saved_without_validation(tmp_path):
    """The rescue (`last`) checkpoint is written at train-epoch end even
    when validation is skipped entirely, which is the whole point of
    decoupling the two callbacks - an HPC job that crashes or times out
    between validations should still have a recent snapshot to resume from.
    """
    train_loader = DataLoader(
        TensorDataset(torch.randn(8, 1), torch.randn(8, 1)), batch_size=2
    )

    ckpt_dir = Path(tmp_path) / "checkpoints"
    val_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="min_val_loss",
        monitor="val_mean_loss",
        mode="min",
        save_top_k=1,
        save_on_train_epoch_end=False,
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
