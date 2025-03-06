# Standard library
from unittest.mock import MagicMock, patch

# Third-party
import torch

# First-party
import neural_lam


def test_trainer_instantiates():
    neural_lam.trainer.Trainer()


def test_configure_optimizer_factory(model):
    trainer = neural_lam.trainer.Trainer(scheduler_config="graphcast")
    configure_optimizers = trainer.get_configure_optimizers_callback()
    [optimizer], [scheduler] = configure_optimizers(model)

    assert isinstance(optimizer, torch.optim.Adam)
    assert isinstance(
        scheduler, neural_lam.lr_scheduler.WarmupCosineAnnealingLR
    )


def test_model_can_configure_optimizers(model):
    trainer = neural_lam.trainer.Trainer(scheduler_config="graphcast")
    with patch.object(
        neural_lam.trainer.Trainer.__bases__[0], "fit", MagicMock()
    ):
        trainer.fit(model)

    [optimizer], [scheduler] = model.configure_optimizers()

    assert isinstance(optimizer, torch.optim.Adam)
    assert isinstance(
        scheduler, neural_lam.lr_scheduler.WarmupCosineAnnealingLR
    )


def test_can_instantiate_torch_scheduler(optimizer):
    scheduler_config = {
        "scheduler": "StepLR",
        "kwargs": {"step_size": 10, "gamma": 0.1},
    }

    scheduler = neural_lam.trainer.get_scheduler(optimizer, scheduler_config)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert scheduler.step_size == 10
    assert scheduler.gamma == 0.1
