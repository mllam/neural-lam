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


# Tests for validate_config
def test_validate_config_passes_with_existing_datastore_path(tmp_path):
    """validate_config should not raise when datastore config_path exists."""
    datastore_file = tmp_path / "datastore.yaml"
    datastore_file.write_text("dummy: true\n")

    nlam_config_path = str(tmp_path / "nlam_config.yaml")

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind="mdp", config_path="datastore.yaml"
        ),
        training=nlconfig.TrainingConfig(),
    )

    nlconfig.validate_config(config, nlam_config_path)


def test_validate_config_raises_on_missing_datastore_file(tmp_path):
    """validate_config raises InvalidConfigError when the resolved
    datastore config path does not exist on disk."""
    nlam_config_path = str(tmp_path / "nlam_config.yaml")

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind="mdp", config_path="does_not_exist.yaml"
        ),
        training=nlconfig.TrainingConfig(),
    )

    with pytest.raises(nlconfig.InvalidConfigError, match="datastore.config_path"):
        nlconfig.validate_config(config, nlam_config_path)


def test_validate_config_error_message_contains_resolved_path(tmp_path):
    """The error message must contain the resolved path so users
    know exactly what file is missing."""
    nlam_config_path = str(tmp_path / "nlam_config.yaml")

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind="mdp", config_path="missing.yaml"
        ),
        training=nlconfig.TrainingConfig(),
    )

    with pytest.raises(nlconfig.InvalidConfigError) as exc_info:
        nlconfig.validate_config(config, nlam_config_path)

    assert "missing.yaml" in str(exc_info.value)


def test_validate_config_raises_on_empty_manual_weights(tmp_path):
    """ManualStateFeatureWeighting with an empty weights dict is invalid
    and should raise InvalidConfigError at startup."""
    datastore_file = tmp_path / "datastore.yaml"
    datastore_file.write_text("dummy: true\n")
    nlam_config_path = str(tmp_path / "nlam_config.yaml")

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind="mdp", config_path="datastore.yaml"
        ),
        training=nlconfig.TrainingConfig(
            state_feature_weighting=nlconfig.ManualStateFeatureWeighting(
                weights={}
            )
        ),
    )

    with pytest.raises(
        nlconfig.InvalidConfigError, match="state_feature_weighting"
    ):
        nlconfig.validate_config(config, nlam_config_path)


def test_validate_config_passes_with_manual_weights(tmp_path):
    """ManualStateFeatureWeighting with actual weights should pass."""
    datastore_file = tmp_path / "datastore.yaml"
    datastore_file.write_text("dummy: true\n")
    nlam_config_path = str(tmp_path / "nlam_config.yaml")

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind="mdp", config_path="datastore.yaml"
        ),
        training=nlconfig.TrainingConfig(
            state_feature_weighting=nlconfig.ManualStateFeatureWeighting(
                weights={"u100m": 1.0, "v100m": 0.5}
            )
        ),
    )

    nlconfig.validate_config(config, nlam_config_path)
