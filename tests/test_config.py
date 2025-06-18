# Third-party
import pytest

# First-party
import neural_lam.config as nlconfig


@pytest.mark.parametrize(
    "state_weighting_config",
    [
        nlconfig.ManualStateFeatureWeighting(
            weights=dict(u100m=1.0, v100m=0.5)
        ),
        nlconfig.UniformFeatureWeighting(),
    ],
)
def test_config_serialization(state_weighting_config):
    c = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(kind="mdp", config_path=""),
        training=nlconfig.TrainingConfig(
            state_feature_weighting=state_weighting_config
        ),
    )

    assert c == c.from_json(c.to_json())
    assert c == c.from_yaml(c.to_yaml())


yaml_training_defaults = """
datastore:
  kind: mdp
  config_path: ""
"""

default_config = nlconfig.NeuralLAMConfig(
    datastore=nlconfig.DatastoreSelection(kind="mdp", config_path=""),
    training=nlconfig.TrainingConfig(
        state_feature_weighting=nlconfig.UniformFeatureWeighting()
    ),
)

yaml_training_manual_weights = """
datastore:
  kind: mdp
  config_path: ""
training:
  state_feature_weighting:
    __config_class__: ManualStateFeatureWeighting
    weights:
      u100m: 1.0
      v100m: 1.0
"""

manual_weights_config = nlconfig.NeuralLAMConfig(
    datastore=nlconfig.DatastoreSelection(kind="mdp", config_path=""),
    training=nlconfig.TrainingConfig(
        state_feature_weighting=nlconfig.ManualStateFeatureWeighting(
            weights=dict(u100m=1.0, v100m=1.0)
        )
    ),
)

yaml_samples = zip(
    [yaml_training_defaults, yaml_training_manual_weights],
    [default_config, manual_weights_config],
)


@pytest.mark.parametrize("yaml_str, config_expected", yaml_samples)
def test_config_load_from_yaml(yaml_str, config_expected):
    c = nlconfig.NeuralLAMConfig.from_yaml(yaml_str)
    assert c == config_expected


def test_config_load_lr_scheduler():
    yaml_str = """
    datastore:
      kind: mdp
      config_path: ""
    training:
      optimization:
        lr_scheduler: CosineAnnealingLR
        lr_scheduler_kwargs:
          T_max: 1000
          eta_min: 0.0
    """
    c = nlconfig.NeuralLAMConfig.from_yaml(yaml_str)

    assert c.training.optimization.lr_scheduler == "CosineAnnealingLR"
    assert c.training.optimization.lr_scheduler_kwargs == dict(
        T_max=1000, eta_min=0.0
    )


def test_config_load_invalid_lr_scheduler_fails():
    yaml_str = """
    datastore:
      kind: mdp
      config_path: ""
    training:
      optimization:
        lr_scheduler: invalid_StepLR
        lr_scheduler_kwargs:
          step_size: 100
          gamma: 0.5
    """
    with pytest.raises(nlconfig.InvalidConfigError):
        nlconfig.NeuralLAMConfig.from_yaml(yaml_str)
