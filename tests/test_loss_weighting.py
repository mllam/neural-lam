"""Unit tests for neural_lam/loss_weighting.py."""

# Standard library
from unittest.mock import MagicMock

# Third-party
import pytest

# First-party
from neural_lam.config import (
    ManualStateFeatureWeighting,
    UniformFeatureWeighting,
)
from neural_lam.loss_weighting import (
    get_manual_state_feature_weights,
    get_state_feature_weighting,
    get_uniform_state_feature_weights,
)
from tests.dummy_datastore import DummyDatastore


class MockDatastore:
    """Mock datastore providing custom state feature names."""

    def __init__(self, state_features: list[str]) -> None:
        self.state_features = state_features

    def get_vars_names(self, category: str = "state") -> list[str]:
        if category == "state":
            return list(self.state_features)
        return []


def test_manual_weights_valid():
    """Test get_manual_state_feature_weights with valid weights."""
    datastore = MockDatastore(state_features=["temp", "humidity", "wind"])
    config = ManualStateFeatureWeighting(
        weights={"humidity": 2.5, "wind": 0.5, "temp": 1.0}
    )
    weights = get_manual_state_feature_weights(config, datastore)
    assert weights == [1.0, 2.5, 0.5]


def test_manual_weights_with_dummy_datastore():
    """Test get_manual_state_feature_weights with DummyDatastore."""
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
    datastore = MockDatastore(state_features=["temp", "humidity", "wind"])
    config = ManualStateFeatureWeighting(weights={"temp": 1.0, "humidity": 2.0})
    with pytest.raises(ValueError, match="are missing"):
        get_manual_state_feature_weights(config, datastore)


def test_manual_weights_extra_features_raises():
    """Test that extra features in weights raises ValueError."""
    datastore = MockDatastore(state_features=["temp", "humidity"])
    config = ManualStateFeatureWeighting(
        weights={"temp": 1.0, "humidity": 2.0, "extra": 3.0}
    )
    with pytest.raises(ValueError, match="which are not in the datastore"):
        get_manual_state_feature_weights(config, datastore)


@pytest.mark.parametrize(
    "invalid_weight",
    [0.0, -0.0, -1.0, -100.5, float("nan"), float("inf"), float("-inf")],
)
def test_manual_weights_invalid_values_raises(invalid_weight):
    """Test that non-positive or non-finite weights raise ValueError."""
    datastore = MockDatastore(state_features=["feat_a", "feat_b"])
    config = ManualStateFeatureWeighting(
        weights={"feat_a": 1.0, "feat_b": invalid_weight}
    )
    with pytest.raises(
        ValueError,
        match="must be a finite, strictly positive number",
    ):
        get_manual_state_feature_weights(config, datastore)


def test_uniform_weights_valid():
    """Test get_uniform_state_feature_weights with valid features."""
    datastore = MockDatastore(state_features=["a", "b", "c", "d"])
    weights = get_uniform_state_feature_weights(datastore)
    assert len(weights) == 4
    assert weights == [0.25, 0.25, 0.25, 0.25]


def test_uniform_weights_with_dummy_datastore():
    """Test get_uniform_state_feature_weights with DummyDatastore."""
    datastore = DummyDatastore()
    state_names = datastore.get_vars_names(category="state")
    weights = get_uniform_state_feature_weights(datastore)
    assert len(weights) == len(state_names)
    assert weights == [1.0 / len(state_names)] * len(state_names)


def test_uniform_weights_empty_features_raises():
    """Test uniform state weights raise ValueError when features are empty."""
    datastore = MockDatastore(state_features=[])
    with pytest.raises(
        ValueError,
        match="Cannot compute uniform state feature weights",
    ):
        get_uniform_state_feature_weights(datastore)


def test_get_state_feature_weighting_manual():
    """Test get_state_feature_weighting dispatches to manual weighting."""
    datastore = MockDatastore(state_features=["a", "b"])
    manual_config = ManualStateFeatureWeighting(weights={"a": 1.5, "b": 3.5})
    config = MagicMock()
    config.training.state_feature_weighting = manual_config

    weights = get_state_feature_weighting(config, datastore)
    assert weights == [1.5, 3.5]


def test_get_state_feature_weighting_uniform():
    """Test get_state_feature_weighting dispatches to uniform weighting."""
    datastore = MockDatastore(state_features=["a", "b"])
    uniform_config = UniformFeatureWeighting()
    config = MagicMock()
    config.training.state_feature_weighting = uniform_config

    weights = get_state_feature_weighting(config, datastore)
    assert weights == [0.5, 0.5]


def test_get_state_feature_weighting_unsupported_raises():
    """Test that unsupported weighting config raises NotImplementedError."""
    datastore = MockDatastore(state_features=["a", "b"])
    config = MagicMock()
    config.training.state_feature_weighting = "unsupported_scheme"

    with pytest.raises(
        NotImplementedError,
        match="Unsupported state feature weighting configuration",
    ):
        get_state_feature_weighting(config, datastore)
