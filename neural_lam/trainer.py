# Standard library
from types import MethodType

# Third-party
import pytorch_lightning as pl
import torch


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
            if self.optimizer_config is None:
                optimizer = get_default_optimizer(pl_module)

            if self.scheduler_config is not None:
                scheduler = get_scheduler(self.scheduler_config, optimizer)
                return [optimizer], [scheduler]

            return optimizer

        return configure_optimizers


def get_optimizer(optimizer_config, pl_module):
    if optimizer_config is None:
        return get_default_optimizer(pl_module)

    # TODO use this code when optimizer_config can be passed as a dict
    # optimizer_cls = getattr(torch.optim, optimizer_config["optimizer"])
    # return optimizer_cls(pl_module.parameters(), **optimizer_config["kwargs"])
    optimizer_cls = getattr(torch.optim, optimizer_config)
    return optimizer_cls(pl_module.parameters())


def get_default_optimizer(pl_module):
    return torch.optim.AdamW(
        pl_module.parameters(), lr=0.001, betas=(0.9, 0.95)
    )


def get_scheduler(optimizer, scheduler_config):
    # TODO use this code when WarmupCosineAnnealingLR is implemented
    # scheduler_name = scheduler_config["scheduler"]
    # if scheduler_name == "WarmupCosineAnnealingLR":
    #     scheduler_cls = neural_lam.lr_scheduler.WarmupCosineAnnealingLR
    # else:
    #     scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)

    # TODO use this code when scheduler_config can be passed as a dict
    #  scheduler_cls = getattr(
    #      torch.optim.lr_scheduler, scheduler_config["scheduler"]
    #  )
    #  return scheduler_cls(optimizer, **scheduler_config["kwargs"])

    scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_config)
    return scheduler_cls(optimizer)
