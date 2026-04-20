# Standard library
from pathlib import Path

# Third-party
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.datastore import DATASTORES
from neural_lam.models.graph_lam import GraphLAM

# Local
from tests.dummy_datastore import DummyDatastore

# Register DummyDatastore so DatastoreSelection validation passes
DATASTORES.setdefault(DummyDatastore.SHORT_NAME, DummyDatastore)


def test_clamping():
    datastore = DummyDatastore()

    graph_name = "1level"

    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name

    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            n_max_levels=1,
        )

    class ModelArgs:
        output_std = False
        loss = "mse"
        restore_opt = False
        n_example_pred = 1
        graph = graph_name
        hidden_dim = 4
        hidden_layers = 1
        processor_layers = 2
        mesh_aggr = "sum"
        lr = 1.0e-3
        val_steps_to_log = [1, 3]
        metrics_watch = []
        num_past_forcing_steps = 1
        num_future_forcing_steps = 1

    model_args = ModelArgs()

    # DummyDatastore has state features: state_feat_0 .. state_feat_4
    # Use the first two for lower clamping and the second for upper clamping
    state_features = datastore.get_vars_names(category="state")
    feat_lower = state_features[0]
    feat_both = state_features[1]
    feat_upper = state_features[2]

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        ),
        training=nlconfig.TrainingConfig(
            output_clamping=nlconfig.OutputClamping(
                lower={feat_lower: 0.0, feat_both: 0.0},
                upper={feat_both: 1.0, feat_upper: 100.0},
            )
        ),
    )

    model = GraphLAM(
        args=model_args,
        datastore=datastore,
        config=config,
    )

    features = datastore.get_vars_names(category="state")
    original_state = torch.zeros(1, 1, len(features))
    zero_delta = original_state.clone()

    # Get a state well within the bounds
    original_state[:, :, model.clamp_lower_upper_idx] = (
        model.sigmoid_lower_lims + model.sigmoid_upper_lims
    ) / 2
    original_state[:, :, model.clamp_lower_idx] = model.softplus_lower_lims + 10
    original_state[:, :, model.clamp_upper_idx] = model.softplus_upper_lims - 10

    # Get a delta that tries to push the state out of bounds
    delta = torch.ones_like(zero_delta)
    delta[:, :, model.clamp_lower_upper_idx] = (
        model.sigmoid_upper_lims - model.sigmoid_lower_lims
    ) / 3
    delta[:, :, model.clamp_lower_idx] = -5
    delta[:, :, model.clamp_upper_idx] = 5

    # Check that a delta of 0 gives unchanged state
    zero_prediction = model.get_clamped_new_state(zero_delta, original_state)
    assert (abs(original_state - zero_prediction) < 1e-6).all().item()

    # Make predictions towards bounds for each feature
    prediction = zero_prediction.clone()
    n_loops = 100
    for i in range(n_loops):
        prediction = model.get_clamped_new_state(delta, prediction)

    # check that unclamped states are as expected
    # delta is 1, so they should be 1*n_loops
    assert (
        (
            abs(
                prediction[
                    :,
                    :,
                    list(
                        set(range(len(features)))
                        - set(model.clamp_lower_upper_idx.tolist())
                        - set(model.clamp_lower_idx.tolist())
                        - set(model.clamp_upper_idx.tolist())
                    ),
                ]
                - n_loops
            )
            < 1e-6
        )
        .all()
        .item()
    )

    # Check that clamped states are within bounds
    assert (
        (
            model.sigmoid_lower_lims
            <= prediction[:, :, model.clamp_lower_upper_idx]
            <= model.sigmoid_upper_lims
        )
        .all()
        .item()
    )
    assert (
        (model.softplus_lower_lims <= prediction[:, :, model.clamp_lower_idx])
        .all()
        .item()
    )
    assert (
        (prediction[:, :, model.clamp_upper_idx] <= model.softplus_upper_lims)
        .all()
        .item()
    )

    # Check that prediction is within bounds in original non-normalized space
    unscaled_prediction = prediction * model.state_std + model.state_mean
    features_idx = {f: i for i, f in enumerate(features)}
    lower_lims = {
        features_idx[f]: lim
        for f, lim in config.training.output_clamping.lower.items()
    }
    upper_lims = {
        features_idx[f]: lim
        for f, lim in config.training.output_clamping.upper.items()
    }
    assert (
        (
            torch.tensor(list(lower_lims.values()))
            <= unscaled_prediction[:, :, list(lower_lims.keys())]
        )
        .all()
        .item()
    )
    assert (
        (
            unscaled_prediction[:, :, list(upper_lims.keys())]
            <= torch.tensor(list(upper_lims.values()))
        )
        .all()
        .item()
    )

    # Check that a prediction from a state starting outside the bounds is also
    # pushed within bounds
    invalid_state = original_state + 5 * delta
    assert (
        not (
            model.sigmoid_lower_lims
            <= invalid_state[:, :, model.clamp_lower_upper_idx]
            <= model.sigmoid_upper_lims
        )
        .any()
        .item()
    )
    assert (
        not (
            model.softplus_lower_lims
            <= invalid_state[:, :, model.clamp_lower_idx]
        )
        .any()
        .item()
    )
    assert (
        not (
            invalid_state[:, :, model.clamp_upper_idx]
            <= model.softplus_upper_lims
        )
        .any()
        .item()
    )
    invalid_prediction = model.get_clamped_new_state(zero_delta, invalid_state)
    assert (
        (
            model.sigmoid_lower_lims
            <= invalid_prediction[:, :, model.clamp_lower_upper_idx]
            <= model.sigmoid_upper_lims
        )
        .all()
        .item()
    )
    assert (
        (
            model.softplus_lower_lims
            <= invalid_prediction[:, :, model.clamp_lower_idx]
        )
        .all()
        .item()
    )
    assert (
        (
            invalid_prediction[:, :, model.clamp_upper_idx]
            <= model.softplus_upper_lims
        )
        .all()
        .item()
    )

    # Repeat to check lower sigmoid limit
    prediction = zero_prediction.clone()
    n_loops = 100
    for i in range(n_loops):
        prediction = model.get_clamped_new_state(-delta, prediction)

    assert (
        (
            model.sigmoid_lower_lims
            <= prediction[:, :, model.clamp_lower_upper_idx]
            <= model.sigmoid_upper_lims
        )
        .all()
        .item()
    )

    assert (
        (
            torch.tensor(list(lower_lims.values()))
            <= unscaled_prediction[:, :, list(lower_lims.keys())]
        )
        .all()
        .item()
    )
    assert (
        (
            unscaled_prediction[:, :, list(upper_lims.keys())]
            <= torch.tensor(list(upper_lims.values()))
        )
        .all()
        .item()
    )

    invalid_state = original_state - 5 * delta
    assert (
        not (
            model.sigmoid_lower_lims
            <= invalid_state[:, :, model.clamp_lower_upper_idx]
            <= model.sigmoid_upper_lims
        )
        .any()
        .item()
    )
    invalid_prediction = model.get_clamped_new_state(zero_delta, invalid_state)
    assert (
        (
            model.sigmoid_lower_lims
            <= invalid_prediction[:, :, model.clamp_lower_upper_idx]
            <= model.sigmoid_upper_lims
        )
        .all()
        .item()
    )
