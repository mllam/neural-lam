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
            f"in the datastore ({state_feature_names}). {missing_features}"
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
