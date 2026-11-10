# Standard library
import math

# Third-party
import numpy as np
import pytest
import torch
import xarray as xr

# First-party
from neural_lam import config as nlconfig
from neural_lam.loss_weighting import (
    get_per_var_std,
    get_pressure_level_state_feature_weights,
    get_state_feature_weighting,
)
from tests.conftest import init_datastore_example
from tests.dummy_datastore import DummyDatastore


class PressureDummyDatastore(DummyDatastore):
    """Dummy datastore with configurable state features for testing."""

    def __init__(self, state_features: list[str] | None = None):
        super().__init__()
        self._custom_state_features = (
            list(state_features)
            if state_features is not None
            else ["t500", "t850", "t1000", "z500", "z850", "z1000", "2t"]
        )

    def get_vars_names(self, category: str) -> list[str]:
        if category == "state":
            return list(self._custom_state_features)
        return super().get_vars_names(category=category)

    def get_standardization_dataarray(self, category: str) -> xr.Dataset:
        if category == "state":
            names = self.get_vars_names("state")
            ds = xr.Dataset(coords={"state_feature": names})
            for op in [
                "mean",
                "std",
                "diff_mean_standardized",
                "diff_std_standardized",
            ]:
                ds[f"state_{op}"] = xr.DataArray(
                    np.ones(len(names), dtype=float),
                    coords={"state_feature": names},
                    dims=["state_feature"],
                )
            return ds
        return super().get_standardization_dataarray(category=category)


def _config(datastore, state_feature_weighting):
    """Build a config for the datastore with the given feature weighting."""
    return nlconfig.NeuralLAMConfig(
        datastores={
            "main": nlconfig.DatastoreSelection(
                kind="mdp", config_path=datastore.root_path
            )
        },
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


def test_pressure_level_weighting_weights_and_normalization():
    """Pressure level weights are proportional to pressure coordinate,
    sum to 1 per atmospheric variable, and match other_weights."""
    datastore = PressureDummyDatastore()
    levels = [500, 850, 1000]
    total_p = sum(levels)
    weighting_config = nlconfig.PressureLevelWeighting(
        variables=["t", "z"],
        pressure_levels=levels,
        name_format="{var_name}{level}",
        other_weights={"2t": 1.0},
    )
    config = _config(datastore, weighting_config)

    weights = get_state_feature_weighting(config, datastore)
    expected_weights = [
        500 / total_p,
        850 / total_p,
        1000 / total_p,
        500 / total_p,
        850 / total_p,
        1000 / total_p,
        1.0,
    ]
    assert pytest.approx(weights) == expected_weights
    # Sum of weights across levels for each variable is 1.0
    assert pytest.approx(sum(weights[:3])) == 1.0
    assert pytest.approx(sum(weights[3:6])) == 1.0


def test_per_var_std_pressure_level_weighting():
    """Under unit diff std, per_var_std is 1 / sqrt(weight) for each
    feature."""
    datastore = PressureDummyDatastore()
    levels = [500, 850, 1000]
    total_p = sum(levels)
    weighting_config = nlconfig.PressureLevelWeighting(
        variables=["t", "z"],
        pressure_levels=levels,
        name_format="{var_name}{level}",
        other_weights={"2t": 1.0},
    )
    config = _config(datastore, weighting_config)

    per_var_std = get_per_var_std(config=config, datastore=datastore)
    expected_weights = [
        500 / total_p,
        850 / total_p,
        1000 / total_p,
        500 / total_p,
        850 / total_p,
        1000 / total_p,
        1.0,
    ]
    expected_std = torch.tensor(
        [1.0 / math.sqrt(w) for w in expected_weights], dtype=torch.float32
    )
    torch.testing.assert_close(per_var_std, expected_std)


def test_pressure_level_weighting_closest_name_suggestion():
    """Datastore with float names like 't500.0' gives helpful error with
    closest matches when integer 't500' is configured."""
    datastore = PressureDummyDatastore(
        state_features=["t500.0", "t850.0", "2t"]
    )
    weighting_config = nlconfig.PressureLevelWeighting(
        variables=["t"],
        pressure_levels=[500, 850],
        name_format="{var_name}{level}",
        other_weights={"2t": 1.0},
    )
    with pytest.raises(ValueError, match="closest in datastore: 't500.0'"):
        get_pressure_level_state_feature_weights(weighting_config, datastore)


def test_pressure_level_weighting_missing_features():
    """Datastore features missing from config raise a ValueError."""
    datastore = PressureDummyDatastore()
    weighting_config = nlconfig.PressureLevelWeighting(
        variables=["t"],
        pressure_levels=[500, 850, 1000],
        name_format="{var_name}{level}",
        other_weights={"2t": 1.0},
    )
    with pytest.raises(ValueError, match=r"Missing features:.*z500"):
        get_pressure_level_state_feature_weights(weighting_config, datastore)


def test_pressure_level_weighting_unrecognized_features():
    """Extra features in other_weights raise a ValueError."""
    datastore = PressureDummyDatastore()
    weighting_config = nlconfig.PressureLevelWeighting(
        variables=["t", "z"],
        pressure_levels=[500, 850, 1000],
        name_format="{var_name}{level}",
        other_weights={"2t": 1.0, "unknown_var": 0.5},
    )
    with pytest.raises(
        ValueError,
        match=r"Unrecognized features not in datastore:.*unknown_var",
    ):
        get_pressure_level_state_feature_weights(weighting_config, datastore)


def test_pressure_level_weighting_duplicate_feature():
    """A variable defined both in pressure_levels and other_weights
    raises ValueError."""
    datastore = PressureDummyDatastore()
    weighting_config = nlconfig.PressureLevelWeighting(
        variables=["t", "z"],
        pressure_levels=[500, 850, 1000],
        name_format="{var_name}{level}",
        other_weights={"t500": 1.0, "2t": 1.0},
    )
    with pytest.raises(
        ValueError,
        match="Feature 't500' is defined both as a pressure-level",
    ):
        get_pressure_level_state_feature_weights(weighting_config, datastore)


def test_pressure_level_weighting_empty_variables_or_levels():
    """Empty pressure_levels with variables or vice-versa raises ValueError."""
    datastore = PressureDummyDatastore()
    cfg1 = nlconfig.PressureLevelWeighting(
        variables=["t"], pressure_levels=[], other_weights={"2t": 1.0}
    )
    with pytest.raises(
        ValueError,
        match="pressure_levels cannot be empty when variables are specified",
    ):
        get_pressure_level_state_feature_weights(cfg1, datastore)

    cfg2 = nlconfig.PressureLevelWeighting(
        variables=[], pressure_levels=[500, 850], other_weights={"2t": 1.0}
    )
    with pytest.raises(
        ValueError,
        match="variables cannot be empty when pressure_levels are specified",
    ):
        get_pressure_level_state_feature_weights(cfg2, datastore)


@pytest.mark.parametrize(
    "invalid_level", [0.0, -500.0, float("nan"), float("inf")]
)
def test_pressure_level_weighting_invalid_pressure_levels(invalid_level):
    """Non-positive or non-finite pressure level raises ValueError."""
    datastore = PressureDummyDatastore()
    weighting_config = nlconfig.PressureLevelWeighting(
        variables=["t"],
        pressure_levels=[500, invalid_level],
        name_format="{var_name}{level}",
        other_weights={"2t": 1.0},
    )
    with pytest.raises(
        ValueError,
        match="Pressure level must be a finite, strictly positive number",
    ):
        get_pressure_level_state_feature_weights(weighting_config, datastore)


@pytest.mark.parametrize(
    "invalid_weight", [0.0, -1.0, float("nan"), float("inf")]
)
def test_pressure_level_weighting_invalid_other_weights(invalid_weight):
    """Non-positive or non-finite weight in other_weights raises ValueError."""
    datastore = PressureDummyDatastore()
    weighting_config = nlconfig.PressureLevelWeighting(
        variables=["t", "z"],
        pressure_levels=[500, 850, 1000],
        name_format="{var_name}{level}",
        other_weights={"2t": invalid_weight},
    )
    with pytest.raises(
        ValueError,
        match="Weight for feature '2t' must be a finite, strictly positive",
    ):
        get_pressure_level_state_feature_weights(weighting_config, datastore)
