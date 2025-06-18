# Third-party
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam.models.ar_model import ARModel


class ARModelWithParams(ARModel):
    def __init__(self, args, datastore, config):
        super().__init__(args=args, datastore=datastore, config=config)
        self.layer = torch.nn.Linear(1, 1)


def test_lr_scheduler_reduces_lr(model_args, datastore):
    yaml_str = """
    datastore:
      kind: mdp
      config_path: ""
    training:
      optimization:
        lr: 1
        lr_scheduler: ExponentialLR
        lr_scheduler_kwargs:
            gamma: 0.5
    """
    config = nlconfig.NeuralLAMConfig.from_yaml(yaml_str)

    model = ARModelWithParams(
        args=model_args, datastore=datastore, config=config
    )
    result = model.configure_optimizers()

    optimizer = result["optimizer"]
    lr_scheduler = result["lr_scheduler"]

    assert optimizer.param_groups[0]["lr"] == 1
    lr_scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.5
    lr_scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.25
