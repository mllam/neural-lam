# Standard library
from pathlib import Path

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
        datastores={
            "main": nlconfig.DatastoreSelection(kind="mdp", config_path="")
        },
        training=nlconfig.TrainingConfig(
            state_feature_weighting=state_weighting_config
        ),
    )

    assert c == c.from_json(c.to_json())
    assert c == c.from_yaml(c.to_yaml())


def test_plotting_config_boundary_datastore_roundtrip():
    c = nlconfig.NeuralLAMConfig(
        datastores={
            "danra": nlconfig.DatastoreSelection(kind="mdp", config_path=""),
            "era5": nlconfig.DatastoreSelection(kind="mdp", config_path=""),
        },
        plotting=nlconfig.PlottingConfig(boundary_datastore="era5"),
    )

    assert c.plotting.boundary_datastore == "era5"
    assert c == c.from_yaml(c.to_yaml())


yaml_training_defaults = """
datastores:
  main:
    kind: mdp
    config_path: ""
"""

default_config = nlconfig.NeuralLAMConfig(
    datastores={
        "main": nlconfig.DatastoreSelection(kind="mdp", config_path="")
    },
    training=nlconfig.TrainingConfig(
        state_feature_weighting=nlconfig.UniformFeatureWeighting()
    ),
)

yaml_training_manual_weights = """
datastores:
  main:
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
    datastores={
        "main": nlconfig.DatastoreSelection(kind="mdp", config_path="")
    },
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


def test_legacy_datastore_key_raises_migration_error(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "datastore:\n  kind: mdp\n  config_path: ''\n", encoding="utf-8"
    )
    with pytest.raises(nlconfig.InvalidConfigError, match="datastores:"):
        nlconfig.load_config_and_datastore(str(config_path))


def test_malformed_yaml_raises_invalid_config_error(tmp_path):
    """A syntactically broken config file must raise `InvalidConfigError`
    like every other config problem, not a raw `yaml.YAMLError`."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "datastores:\n  danra:\n  - broken: [", encoding="utf-8"
    )
    with pytest.raises(nlconfig.InvalidConfigError):
        nlconfig.load_config_and_datastore(str(config_path))


DATASTORE_EXAMPLES = Path(__file__).parent / "datastore_examples" / "mdp"
BOUNDARY_EXAMPLE_DIR = DATASTORE_EXAMPLES / "era5_1000hPa_danra_100m_winds"


def _write_config(tmp_path, entries):
    """Write a neural-lam config with the given datastore entries.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory to write the config into.
    entries : dict
        Mapping from datastore name to the datastore config path.

    Returns
    -------
    pathlib.Path
        Path to the written config.
    """
    lines = ["datastores:"]
    for name, path in entries.items():
        lines += [f"  {name}:", "    kind: mdp", f"    config_path: {path}"]
    config_path = tmp_path / "config.yaml"
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


@pytest.mark.slow
def test_load_config_and_datastore_splits_by_state(monkeypatch, tmp_path):
    """The shipped two-datastore example loads into (config, interior,
    boundary), with the roles decided by which datastore has `state` data,
    and `config_path` entries resolved relative to the config file rather
    than the working directory."""
    # Run from elsewhere: the entries are relative, so a CWD-relative
    # resolution would not find them.
    monkeypatch.chdir(tmp_path)

    config, datastore, datastore_boundary = nlconfig.load_config_and_datastore(
        str(BOUNDARY_EXAMPLE_DIR / "config.yaml")
    )

    assert set(config.datastores) == {"danra", "era5"}
    assert datastore.get_num_data_vars("state") > 0
    assert datastore_boundary is not None
    assert datastore_boundary.get_num_data_vars("state") == 0
    assert datastore_boundary.get_num_data_vars("forcing") > 0


@pytest.mark.slow
def test_load_config_and_datastore_requires_an_interior(tmp_path):
    """A config whose datastores all lack `state` data has no interior."""
    config_path = _write_config(
        tmp_path, {"era5": BOUNDARY_EXAMPLE_DIR / "era5.datastore.yaml"}
    )

    with pytest.raises(
        nlconfig.InvalidConfigError, match="Exactly one datastore must provide"
    ):
        nlconfig.load_config_and_datastore(str(config_path))


@pytest.mark.slow
def test_load_config_and_datastore_rejects_two_interiors(tmp_path):
    """Two datastores with `state` data are ambiguous as the interior."""
    danra = DATASTORE_EXAMPLES / "danra_100m_winds" / "danra.datastore.yaml"
    config_path = _write_config(tmp_path, {"a": danra, "b": danra})

    with pytest.raises(
        nlconfig.InvalidConfigError, match="Exactly one datastore must provide"
    ):
        nlconfig.load_config_and_datastore(str(config_path))


@pytest.mark.slow
def test_load_config_and_datastore_rejects_two_boundaries(tmp_path):
    """Only a single boundary datastore is supported for now."""
    danra = DATASTORE_EXAMPLES / "danra_100m_winds" / "danra.datastore.yaml"
    era5 = BOUNDARY_EXAMPLE_DIR / "era5.datastore.yaml"
    config_path = _write_config(
        tmp_path, {"interior": danra, "b1": era5, "b2": era5}
    )

    with pytest.raises(
        nlconfig.InvalidConfigError, match="At most one boundary datastore"
    ):
        nlconfig.load_config_and_datastore(str(config_path))
