# Standard library
import math

# Third-party
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam.loss_weighting import get_per_var_std
from tests.conftest import init_datastore_example
from tests.dummy_datastore import DummyDatastore


def _config(datastore, state_feature_weighting):
    """Build a config for the datastore with the given feature weighting."""
    return nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        ),
        training=nlconfig.TrainingConfig(
            state_feature_weighting=state_feature_weighting
        ),
    )


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
