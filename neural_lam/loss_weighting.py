"""Utility functions for configuring state-feature loss weighting."""

# Standard library
import difflib
import math

# Third-party
import torch

# Local
from .config import (
    ManualStateFeatureWeighting,
    NeuralLAMConfig,
    PressureLevelWeighting,
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


def get_pressure_level_state_feature_weights(
    weighting_config: PressureLevelWeighting, datastore: BaseDatastore
) -> list[float]:
    """
    Return the state feature weights as a list of floats in the order of the
    state features in the datastore, with vertical atmospheric variables
    weighted proportionally to pressure levels and other features assigned
    explicit manual weights.

    Parameters
    ----------
    weighting_config : PressureLevelWeighting
        Configuration object containing pressure-level weighting settings.
    datastore : BaseDatastore
        Datastore object containing the state features.

    Returns
    -------
    list[float]
        List of floats containing the state feature weights.

    Raises
    ------
    ValueError
        If ``pressure_levels`` contains non-positive or non-finite values,
        if sum of pressure levels is non-positive or non-finite,
        if ``other_weights`` contains non-positive or non-finite values,
        if a feature name is duplicated, or if the configured features do not
        match the state features in the datastore.
    """
    if weighting_config.variables:
        if not weighting_config.pressure_levels:
            raise ValueError(
                "pressure_levels cannot be empty when variables are specified."
            )
        for level in weighting_config.pressure_levels:
            if not math.isfinite(level) or level <= 0:
                raise ValueError(
                    f"Pressure level must be a finite, strictly positive "
                    f"number (level > 0), got {level}."
                )
        total_pressure = float(sum(weighting_config.pressure_levels))
        if not math.isfinite(total_pressure) or total_pressure <= 0:
            raise ValueError(
                "Sum of pressure levels must be a finite, strictly "
                "positive number."
            )
    elif weighting_config.pressure_levels:
        raise ValueError(
            "variables cannot be empty when pressure_levels are specified."
        )

    for feat, w in weighting_config.other_weights.items():
        if not math.isfinite(w) or w <= 0:
            raise ValueError(
                f"Weight for feature '{feat}' must be a finite, strictly "
                f"positive number (weight > 0), got {w}."
            )

    computed_weights: dict[str, float] = {}
    if weighting_config.variables:
        total_pressure = float(sum(weighting_config.pressure_levels))
        for var_name in weighting_config.variables:
            for level in weighting_config.pressure_levels:
                feat_name = weighting_config.name_format.format(
                    var_name=var_name, level=level
                )
                if feat_name in computed_weights:
                    raise ValueError(
                        f"Feature '{feat_name}' is defined multiple times."
                    )
                computed_weights[feat_name] = float(level) / total_pressure

    for feat, w in weighting_config.other_weights.items():
        if feat in computed_weights:
            raise ValueError(
                f"Feature '{feat}' is defined both as a pressure-level "
                "variable and in other_weights."
            )
        computed_weights[feat] = float(w)

    state_feature_names = datastore.get_vars_names(category="state")
    if set(computed_weights.keys()) != set(state_feature_names):
        missing = sorted(
            set(state_feature_names) - set(computed_weights.keys())
        )
        unrecognized = sorted(
            set(computed_weights.keys()) - set(state_feature_names)
        )
        error_parts = [
            "PressureLevelWeighting features do not match state features "
            "in the datastore."
        ]
        if missing:
            error_parts.append(f"Missing features: {missing}.")
        if unrecognized:
            suggestions = []
            candidates = missing if missing else state_feature_names
            for feat in unrecognized:
                if f"{feat}.0" in candidates:
                    match = f"{feat}.0"
                elif feat.endswith(".0") and feat[:-2] in candidates:
                    match = feat[:-2]
                else:
                    matches = difflib.get_close_matches(
                        feat, candidates, n=1, cutoff=0.5
                    )
                    match = matches[0] if matches else None

                if match:
                    suggestions.append(
                        f"{feat} (closest in datastore: '{match}')"
                    )
                else:
                    suggestions.append(feat)
            error_parts.append(
                f"Unrecognized features not in datastore: {suggestions}."
            )
        raise ValueError(" ".join(error_parts))

    return [computed_weights[name] for name in state_feature_names]


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
    elif isinstance(weighting_config, PressureLevelWeighting):
        weights = get_pressure_level_state_feature_weights(
            weighting_config, datastore
        )
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
