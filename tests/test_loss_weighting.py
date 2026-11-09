"""Unit tests for neural_lam/loss_weighting.py."""

# Standard library
import math

# Third-party
import pytest
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam.config import (
    ManualStateFeatureWeighting,
    UniformFeatureWeighting,
)
from neural_lam.loss_weighting import (
    get_manual_state_feature_weights,
    get_per_var_std,
    get_state_feature_weighting,
    get_uniform_state_feature_weights,
)
from tests.conftest import init_datastore_example
from tests.dummy_datastore import DummyDatastore


class EmptyStateDatastore(DummyDatastore):
    """DummyDatastore subclass returning no state features."""

    def get_vars_names(self, category: str = "state") -> list[str]:
        if category == "state":
            return []
        return super().get_vars_names(category=category)


def _config(datastore, state_feature_weighting):
    """Build a config for the datastore with the given feature weighting."""
    return nlconfig.NeuralLAMConfig(
        datastores={
            "main": nlconfig.DatastoreSelection(
                kind=datastore.SHORT_NAME, config_path=datastore.root_path
            )
        },
        training=nlconfig.TrainingConfig(
            state_feature_weighting=state_feature_weighting
        ),
    )


def test_manual_weights_valid():
    """Test get_manual_state_feature_weights with valid weights."""
    datastore = DummyDatastore()
    state_names = datastore.get_vars_names(category="state")
    expected_weights = {
        name: float(i + 1) for i, name in enumerate(state_names)
    }
    config = ManualStateFeatureWeighting(weights=expected_weights)
    weights = get_manual_state_feature_weights(config, datastore)
    assert weights == [float(i + 1) for i in range(len(state_names))]


def test_manual_weights_missing_features_raises():
    """Test that missing state features in weights raises ValueError."""
    datastore = DummyDatastore()
    state_names = datastore.get_vars_names(category="state")
    config = ManualStateFeatureWeighting(
        weights={name: 1.0 for name in state_names[:-1]}
    )
    with pytest.raises(ValueError, match="are missing"):
        get_manual_state_feature_weights(config, datastore)


def test_manual_weights_extra_features_raises():
    """Test that extra features in weights raises ValueError."""
    datastore = DummyDatastore()
    state_names = datastore.get_vars_names(category="state")
    weights = {name: 1.0 for name in state_names}
    weights["extra_nonexistent_feature"] = 2.0
    config = ManualStateFeatureWeighting(weights=weights)
    with pytest.raises(ValueError, match="which are not in the datastore"):
        get_manual_state_feature_weights(config, datastore)


@pytest.mark.parametrize(
    "invalid_weight",
    [0.0, -0.0, -1.0, -100.5, float("nan"), float("inf"), float("-inf")],
)
def test_manual_weights_invalid_values_raises(invalid_weight):
    """Test that non-positive or non-finite weights raise ValueError."""
    datastore = DummyDatastore()
    state_names = datastore.get_vars_names(category="state")
    weights = {name: 1.0 for name in state_names}
    weights[state_names[0]] = invalid_weight
    config = ManualStateFeatureWeighting(weights=weights)
    with pytest.raises(
        ValueError,
        match="must be a finite, strictly positive number",
    ):
        get_manual_state_feature_weights(config, datastore)


def test_uniform_weights_valid():
    """Test get_uniform_state_feature_weights with DummyDatastore."""
    datastore = DummyDatastore()
    state_names = datastore.get_vars_names(category="state")
    weights = get_uniform_state_feature_weights(datastore)
    assert len(weights) == len(state_names)
    assert weights == [1.0 / len(state_names)] * len(state_names)


def test_uniform_weights_empty_features_raises():
    """Test uniform state weights raise ValueError when features are empty."""
    datastore = EmptyStateDatastore()
    with pytest.raises(
        ValueError,
        match="Cannot compute uniform state feature weights",
    ):
        get_uniform_state_feature_weights(datastore)


def test_get_state_feature_weighting_manual():
    """Test get_state_feature_weighting dispatches to manual weighting."""
    datastore = DummyDatastore()
    state_names = datastore.get_vars_names(category="state")
    manual_weights = {name: float(i + 1) for i, name in enumerate(state_names)}
    config = _config(
        datastore,
        ManualStateFeatureWeighting(weights=manual_weights),
    )

    weights = get_state_feature_weighting(config, datastore)
    assert weights == [float(i + 1) for i in range(len(state_names))]


def test_get_state_feature_weighting_uniform():
    """Test get_state_feature_weighting dispatches to uniform weighting."""
    datastore = DummyDatastore()
    state_names = datastore.get_vars_names(category="state")
    config = _config(datastore, UniformFeatureWeighting())

    weights = get_state_feature_weighting(config, datastore)
    assert weights == [1.0 / len(state_names)] * len(state_names)


def test_get_state_feature_weighting_unsupported_raises():
    """Test that unsupported weighting config raises NotImplementedError."""
    datastore = DummyDatastore()
    config = _config(datastore, "unsupported_scheme")

    with pytest.raises(
        NotImplementedError,
        match="Unsupported state feature weighting configuration",
    ):
        get_state_feature_weighting(config, datastore)


def test_per_var_std_divides_by_sqrt_of_manual_weights():
    """The dummy datastore has a unit difference std for every feature, so
    the returned std is exactly one over the square root of each weight."""
    datastore = DummyDatastore()
    weights = [1.0, 4.0, 0.25, 1.0, 16.0]
    config = _config(
        datastore,
        nlconfig.ManualStateFeatureWeighting(
            weights=dict(zip(datastore.get_vars_names("state"), weights))
        ),
    )

    per_var_std = get_per_var_std(config=config, datastore=datastore)

    torch.testing.assert_close(
        per_var_std, torch.tensor([1.0, 0.5, 2.0, 1.0, 0.25])
    )


def test_per_var_std_uniform_weighting_scales_by_num_features():
    """Uniform weighting gives every one of the dummy datastore's five
    features a weight of 1/5, and so a std of sqrt(5)."""
    datastore = DummyDatastore()
    config = _config(datastore, nlconfig.UniformFeatureWeighting())

    per_var_std = get_per_var_std(config=config, datastore=datastore)

    torch.testing.assert_close(
        per_var_std, torch.full((5,), math.sqrt(5.0), dtype=torch.float32)
    )


@pytest.mark.requires_real_data
def test_per_var_std_uses_the_difference_std():
    """Under unit weights the std is the standardized one-step-difference
    std, which for a real datastore is not the state std."""
    datastore = init_datastore_example("mdp")
    da_state_stats = datastore.get_standardization_dataarray(category="state")
    config = _config(
        datastore,
        nlconfig.ManualStateFeatureWeighting(
            weights={name: 1.0 for name in datastore.get_vars_names("state")}
        ),
    )

    per_var_std = get_per_var_std(config=config, datastore=datastore)

    torch.testing.assert_close(
        per_var_std,
        torch.tensor(
            da_state_stats.state_diff_std_standardized.values,
            dtype=torch.float32,
        ),
    )
    assert not torch.allclose(
        per_var_std,
        torch.tensor(da_state_stats.state_std.values, dtype=torch.float32),
    )
