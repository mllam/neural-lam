"""Utility functions for configuring state-feature loss weighting."""

# Third-party
import torch

# Local
from .config import (
    ManualStateFeatureWeighting,
    NeuralLAMConfig,
    UniformFeatureWeighting,
)
from .datastore.base import BaseDatastore


def get_manual_state_feature_weights(
    weighting_config: ManualStateFeatureWeighting, datastore: BaseDatastore
) -> list[float]:
    """
    Return the state feature weights as a list of floats in the order of the
    state features in the datastore.

    Parameters
    ----------
    weighting_config : ManualStateFeatureWeighting
        Configuration object containing the manual state feature weights.
    datastore : BaseDatastore
        Datastore object containing the state features.

    Returns
    -------
    list[float]
        List of floats containing the state feature weights.

    Raises
    ------
    ValueError
        If the set of feature names in ``weighting_config.weights`` does
        not match the state feature names in the datastore.
    """
    state_feature_names = datastore.get_vars_names(category="state")
    feature_weight_names = weighting_config.weights.keys()

    # Check that the state_feature_weights dictionary has a weight for each
    # state feature in the datastore.
    if set(feature_weight_names) != set(state_feature_names):
        additional_features = set(feature_weight_names) - set(
            state_feature_names
        )
        missing_features = set(state_feature_names) - set(feature_weight_names)
        raise ValueError(
            f"State feature weights must be provided for each state feature"
            f" in the datastore ({state_feature_names}). {missing_features}"
            " are missing and weights are defined for the features "
            f"{additional_features} which are not in the datastore."
        )

    state_feature_weights = [
        weighting_config.weights[feature] for feature in state_feature_names
    ]
    return state_feature_weights


def get_uniform_state_feature_weights(datastore: BaseDatastore) -> list[float]:
    """
    Return the state feature weights as a list of floats in the order of the
    state features in the datastore.

    The weights are uniform, i.e. 1.0/n_features for each feature.

    Parameters
    ----------
    datastore : BaseDatastore
        Datastore object containing the state features.

    Returns
    -------
    list[float]
        List of floats containing the state feature weights.
    """
    state_feature_names = datastore.get_vars_names(category="state")
    n_features = len(state_feature_names)
    return [1.0 / n_features] * n_features


def get_state_feature_weighting(
    config: NeuralLAMConfig, datastore: BaseDatastore
) -> list[float]:
    """
    Return the state feature weights as a list of floats in the order of the
    state features in the datastore. The weights are determined based on the
    configuration in the NeuralLAMConfig object.

    Parameters
    ----------
    config : NeuralLAMConfig
        Configuration object for neural-lam.
    datastore : BaseDatastore
        Datastore object containing the state features.

    Returns
    -------
    list[float]
        List of floats containing the state feature weights.

    Raises
    ------
    NotImplementedError
        If ``config.training.state_feature_weighting`` is not a
        recognised weighting configuration type.
    """
    weighting_config = config.training.state_feature_weighting

    if isinstance(weighting_config, ManualStateFeatureWeighting):
        weights = get_manual_state_feature_weights(weighting_config, datastore)
    elif isinstance(weighting_config, UniformFeatureWeighting):
        weights = get_uniform_state_feature_weights(datastore)
    else:
        raise NotImplementedError(
            "Unsupported state feature weighting configuration: "
            f"{weighting_config}"
        )

    return weights


def get_per_var_std(
    config: NeuralLAMConfig, datastore: BaseDatastore
) -> torch.Tensor:
    """
    Return the constant per-variable standard deviation of the one-step
    difference, weighted by the configured state feature weighting.

    Forecasters whose predictor does not output its own standard deviation
    substitute this for ``pred_std`` when applying a scoring rule.

    Parameters
    ----------
    config : NeuralLAMConfig
        Configuration object for neural-lam, supplying the state feature
        weighting.
    datastore : BaseDatastore
        Datastore object containing the state standardization statistics.

    Returns
    -------
    torch.Tensor
        Shape ``(num_state_vars,)``. Per-variable standard deviation.
    """
    da_state_stats = datastore.get_standardization_dataarray(category="state")
    diff_std = torch.tensor(
        da_state_stats.state_diff_std_standardized.values,
        dtype=torch.float32,
    )
    feature_weights = torch.tensor(
        get_state_feature_weighting(config=config, datastore=datastore),
        dtype=torch.float32,
    )
    return diff_std / torch.sqrt(feature_weights)
