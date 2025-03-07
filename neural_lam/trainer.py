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

        def configure_optimizers(pl_module):
            optimizer = get_optimizer(self.optimizer_config, pl_module)

            if self.scheduler_config is not None:
                scheduler = get_scheduler(self.scheduler_config, optimizer)
                return [optimizer], [scheduler]

            return optimizer

        return configure_optimizers


def get_optimizer(optimizer_config, model):
    optimizer_cls = getattr(torch.optim, optimizer_config["optimizer"])
    return optimizer_cls(model.parameters(), **optimizer_config["kwargs"])


def get_scheduler(optimizer, scheduler_config):
    # TODO use this code when WarmupCosineAnnealingLR is implemented
    # scheduler_name = scheduler_config["scheduler"]
    # if scheduler_name == "WarmupCosineAnnealingLR":
    #     scheduler_cls = neural_lam.lr_scheduler.WarmupCosineAnnealingLR
    # else:
    #     scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)

    scheduler_cls = getattr(
        torch.optim.lr_scheduler, scheduler_config["scheduler"]
    )

    return scheduler_cls(optimizer, **scheduler_config["kwargs"])
