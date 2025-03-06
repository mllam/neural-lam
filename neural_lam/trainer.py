# Standard library
from types import MethodType

# Third-party
import pytorch_lightning as pl
import torch

# First-party
import neural_lam


class Trainer(pl.Trainer):
    def __init__(
        self, scheduler_config=None, optimizer_config=None, *args, **kwargs
    ):
        self.scheduler_config = scheduler_config
        self.optimizer_config = optimizer_config

        super().__init__(*args, **kwargs)

    def fit(
        self,
        model,
        *args,
        **kwargs,
    ):
        model.configure_optimizers = MethodType(
            self.get_configure_optimizers_callback(), model
        )
        super().fit(model, *args, **kwargs)

    def get_configure_optimizers_callback(self):

        # TODO configure optimizer and scheduler from member config files

        def configure_optimizers(pl_module):
            optimizer = torch.optim.Adam(pl_module.parameters())
            if self.scheduler_config == "graphcast":
                scheduler = neural_lam.lr_scheduler.WarmupCosineAnnealingLR(
                    optimizer,
                )
                return [optimizer], [scheduler]
            return optimizer

        return configure_optimizers


def get_scheduler(optimizer, scheduler_config):
    scheduler = getattr(torch.optim.lr_scheduler, scheduler_config["scheduler"])
    return scheduler(optimizer, **scheduler_config["kwargs"])
