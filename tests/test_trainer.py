# Third-party
import torch

# First-party
import neural_lam


def test_trainer_instantiates():
    neural_lam.trainer.Trainer()


def test_configure_optimizer_factory(model):
    trainer = neural_lam.trainer.Trainer()
    configure_optimizers = trainer.get_configure_optimizers_callback()
    [optimizer], [scheduler] = configure_optimizers(model)

    assert isinstance(optimizer, torch.optim.Adam)
    assert isinstance(
        scheduler, neural_lam.lr_scheduler.WarmupCosineAnnealingLR
    )
