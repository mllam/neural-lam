# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.datastore.mdp import MDPDatastore
from neural_lam.models import GraphLAM
from tests.conftest import init_datastore_example


@pytest.fixture(scope="module")
def clamping_setup():
    """Build a GraphLAM with clamping limits once for the whole module.

    Constructing the datastore, graph and model takes about a second, and
    `@given` re-runs a test body once per generated example, so this is
    built at module scope rather than inside the tests.

    Returns
    -------
    tuple
        The model, the config it was built from, and the state feature
        names.
    """
    datastore = init_datastore_example(MDPDatastore.SHORT_NAME)

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
        ar_steps_eval = 3
        metrics_watch = []
        num_past_forcing_steps = 1
        num_future_forcing_steps = 1

    model_args = ModelArgs()

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        ),
        training=nlconfig.TrainingConfig(
            output_clamping=nlconfig.OutputClamping(
                lower={"t2m": 0.0, "r2m": 0.0},
                upper={"r2m": 1.0, "u100m": 100.0},
            )
        ),
    )

    model = GraphLAM(
        datastore=datastore,
        graph_name=model_args.graph,
        hidden_dim=model_args.hidden_dim,
        hidden_layers=model_args.hidden_layers,
        processor_layers=model_args.processor_layers,
        mesh_aggr=model_args.mesh_aggr,
        num_past_forcing_steps=model_args.num_past_forcing_steps,
        num_future_forcing_steps=model_args.num_future_forcing_steps,
        output_std=model_args.output_std,
        output_clamping_lower=config.training.output_clamping.lower,
        output_clamping_upper=config.training.output_clamping.upper,
    )

    features = datastore.get_vars_names(category="state")

    return model, config, features


def test_clamping(clamping_setup):
    model, config, features = clamping_setup

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
    # they should not be at the bounds but allow it due to numerical precision
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
    # pushed within bounds. 3 delta should be enough to give an initial state
    # out of bounds so 5 is well outside
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

    # Above tests only check the upper sigmoid limit.
    # Repeat to check lower sigmoid limit

    # Make predictions towards bounds for each feature
    prediction = zero_prediction.clone()
    n_loops = 100
    for i in range(n_loops):
        prediction = model.get_clamped_new_state(-delta, prediction)

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

    # Check that prediction is within bounds in original non-normalized space
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
    # pushed within bounds. 3 delta should be enough to give an initial state
    # out of bounds so 5 is well outside
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


# Values are drawn in standardized units, where the model operates. 1e3 is
# far beyond anything seen in practice and fully saturates the clamping
# functions, which is precisely the regime the bounds have to survive.
_ELEMENTS = st.floats(
    min_value=-1e3,
    max_value=1e3,
    width=32,
    allow_nan=False,
    allow_infinity=False,
)

# `derandomize` fixes the drawn examples, so a given commit always runs the
# same inputs. That trades hypothesis' open-ended search for determinism,
# on the reasoning that a property test which can fail on an unrelated PR
# tends to get disabled rather than fixed.
# `deadline=None` because per-example timing of torch ops on shared CI
# runners is unreliable, and the 200 ms default would only add flakiness.
_SETTINGS = settings(deadline=None, derandomize=True)


def _draw_state_and_delta(data, num_features):
    """Draw a `(prev_state, state_delta)` pair sharing one random shape.

    Parameters
    ----------
    data : hypothesis.strategies.DataObject
        The `st.data()` object of the calling test.
    num_features : int
        Size of the trailing state-feature dimension.

    Returns
    -------
    tuple of torch.Tensor
        Two tensors of shape `(batch, grid, num_features)`.
    """
    shape = (
        data.draw(st.integers(min_value=1, max_value=3), label="batch"),
        data.draw(st.integers(min_value=1, max_value=4), label="grid"),
        num_features,
    )
    arrays = hnp.arrays(dtype=np.float32, shape=shape, elements=_ELEMENTS)
    return (
        torch.from_numpy(data.draw(arrays, label="prev_state")),
        torch.from_numpy(data.draw(arrays, label="state_delta")),
    )


@_SETTINGS
@given(data=st.data())
def test_clamped_features_stay_within_bounds(clamping_setup, data):
    """Bounds hold for any previous state and any delta.

    `get_clamped_new_state` applies `f(f^-1(x) + delta)`, and both inverses
    clamp internally, so an out-of-range `prev_state` is a valid input rather
    than something that has to be constructed by hand. That makes the bound
    an unconditional property of the whole input space, not of the particular
    trajectory `test_clamping` walks.
    """
    model, _, features = clamping_setup
    prev_state, state_delta = _draw_state_and_delta(data, len(features))

    new_state = model.get_clamped_new_state(state_delta, prev_state)

    assert torch.isfinite(new_state).all()

    # Each bound is asserted on its own rather than as `lower <= x <= upper`:
    # Python expands a chained comparison to `and`, which raises on a tensor
    # holding more than one value.
    sigmoid_state = new_state[:, :, model.clamp_lower_upper_idx]
    assert (sigmoid_state >= model.sigmoid_lower_lims).all()
    assert (sigmoid_state <= model.sigmoid_upper_lims).all()
    assert (
        new_state[:, :, model.clamp_lower_idx] >= model.softplus_lower_lims
    ).all()
    assert (
        new_state[:, :, model.clamp_upper_idx] <= model.softplus_upper_lims
    ).all()


@_SETTINGS
@given(data=st.data())
def test_unclamped_features_are_a_plain_residual_sum(clamping_setup, data):
    """Features without limits are untouched by clamping.

    Clamping must not leak onto features the config left unbounded; those
    keep the plain `prev_state + state_delta` update, exactly.
    """
    model, _, features = clamping_setup
    prev_state, state_delta = _draw_state_and_delta(data, len(features))

    unclamped_idx = sorted(
        set(range(len(features)))
        - set(model.clamp_lower_upper_idx.tolist())
        - set(model.clamp_lower_idx.tolist())
        - set(model.clamp_upper_idx.tolist())
    )
    assert unclamped_idx, "expected at least one unbounded state feature"

    new_state = model.get_clamped_new_state(state_delta, prev_state)

    assert torch.equal(
        new_state[:, :, unclamped_idx],
        prev_state[:, :, unclamped_idx] + state_delta[:, :, unclamped_idx],
    )
