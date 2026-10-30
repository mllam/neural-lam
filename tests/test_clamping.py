# Standard library
from pathlib import Path

# Third-party
import pytest
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph_with_wmg import create_graph_from_datastore
from neural_lam.datastore.mdp import MDPDatastore
from neural_lam.models import GraphLAM
from neural_lam.models.step_predictors.base import StepPredictor
from tests.conftest import init_datastore_example
from tests.dummy_datastore import DummyDatastore


@pytest.mark.requires_real_data
def test_clamping():
    datastore = init_datastore_example(MDPDatastore.SHORT_NAME)

    graph_name = "1level"

    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name

    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            archetype="keisler",
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
        datastores={
            "main": nlconfig.DatastoreSelection(
                kind=datastore.SHORT_NAME, config_path=datastore.root_path
            )
        },
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


class DummyStepPredictor(StepPredictor):
    """
    Concrete StepPredictor for testing clamping without graph setup.
    """

    def forward(
        self,
        prev_state: torch.Tensor,
        prev_prev_state: torch.Tensor,
        forcing: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return prev_state, None


@pytest.fixture
def dummy_datastore():
    """Create a minimal DummyDatastore for clamping unit tests."""
    return DummyDatastore(n_grid_points=4, n_timesteps=2)


def test_clamping_unknown_features_raises(dummy_datastore):
    """
    Test that specifying limits for non-existent features raises ValueError.
    """
    model_lower = DummyStepPredictor(
        dummy_datastore,
        output_clamping_lower={"unknown_feature": 0.0},
    )
    with pytest.raises(
        ValueError,
        match="State feature limits were provided for unknown features",
    ):
        model_lower.prepare_clamping_params(dummy_datastore)

    model_upper = DummyStepPredictor(
        dummy_datastore,
        output_clamping_upper={"non_existent": 1.0},
    )
    with pytest.raises(
        ValueError,
        match="State feature limits were provided for unknown features",
    ):
        model_upper.prepare_clamping_params(dummy_datastore)


def test_clamping_lower_ge_upper_raises(dummy_datastore):
    """
    Test that lower limit greater than or equal to upper raises ValueError.
    """
    model_gt = DummyStepPredictor(
        dummy_datastore,
        output_clamping_lower={"state_feat_0": 5.0},
        output_clamping_upper={"state_feat_0": 2.0},
    )
    with pytest.raises(
        ValueError,
        match='Invalid clamping limits for feature "state_feat_0"',
    ):
        model_gt.prepare_clamping_params(dummy_datastore)

    model_eq = DummyStepPredictor(
        dummy_datastore,
        output_clamping_lower={"state_feat_0": 3.0},
        output_clamping_upper={"state_feat_0": 3.0},
    )
    with pytest.raises(
        ValueError,
        match='Invalid clamping limits for feature "state_feat_0"',
    ):
        model_eq.prepare_clamping_params(dummy_datastore)


@pytest.mark.parametrize("val", [float("nan"), float("inf"), float("-inf")])
def test_clamping_non_finite_lower_limit_raises(dummy_datastore, val):
    """Test that non-finite lower clamping limits raise ValueError."""
    model = DummyStepPredictor(
        dummy_datastore,
        output_clamping_lower={"state_feat_0": val},
    )
    with pytest.raises(
        ValueError,
        match=(
            "Lower clamping limit for feature 'state_feat_0' must be a "
            "finite number"
        ),
    ):
        model.prepare_clamping_params(dummy_datastore)


@pytest.mark.parametrize("val", [float("nan"), float("inf"), float("-inf")])
def test_clamping_non_finite_upper_limit_raises(dummy_datastore, val):
    """Test that non-finite upper clamping limits raise ValueError."""
    model = DummyStepPredictor(
        dummy_datastore,
        output_clamping_upper={"state_feat_0": val},
    )
    with pytest.raises(
        ValueError,
        match=(
            "Upper clamping limit for feature 'state_feat_0' must be a "
            "finite number"
        ),
    ):
        model.prepare_clamping_params(dummy_datastore)


def test_clamping_infinite_bounds_both_raises(dummy_datastore):
    """Test that infinite bounds for both lower and upper raise ValueError."""
    model = DummyStepPredictor(
        dummy_datastore,
        output_clamping_lower={"state_feat_0": float("-inf")},
        output_clamping_upper={"state_feat_0": float("inf")},
    )
    with pytest.raises(
        ValueError,
        match=(
            "Lower clamping limit for feature 'state_feat_0' must be a "
            "finite number"
        ),
    ):
        model.prepare_clamping_params(dummy_datastore)


@pytest.mark.parametrize("val", ["0.0", None, True, False, [1.0]])
def test_clamping_non_numeric_lower_limit_raises(dummy_datastore, val):
    """Test that non-numeric lower clamping limits raise ValueError."""
    model = DummyStepPredictor(
        dummy_datastore,
        output_clamping_lower={"state_feat_0": val},
    )
    with pytest.raises(
        ValueError,
        match=(
            "Lower clamping limit for feature 'state_feat_0' must be a "
            "finite number"
        ),
    ):
        model.prepare_clamping_params(dummy_datastore)


@pytest.mark.parametrize("val", ["1.0", None, True, False, [1.0]])
def test_clamping_non_numeric_upper_limit_raises(dummy_datastore, val):
    """Test that non-numeric upper clamping limits raise ValueError."""
    model = DummyStepPredictor(
        dummy_datastore,
        output_clamping_upper={"state_feat_0": val},
    )
    with pytest.raises(
        ValueError,
        match=(
            "Upper clamping limit for feature 'state_feat_0' must be a "
            "finite number"
        ),
    ):
        model.prepare_clamping_params(dummy_datastore)


def test_clamping_valid_limits_setup_and_forward(dummy_datastore):
    """Test valid clamping limits setup and get_clamped_new_state forward."""
    model = DummyStepPredictor(
        dummy_datastore,
        output_clamping_lower={"state_feat_0": 0.0, "state_feat_1": -5.0},
        output_clamping_upper={"state_feat_1": 5.0, "state_feat_2": 10.0},
    )
    model.prepare_clamping_params(dummy_datastore)

    # feat 0: lower only -> softplus_lower (clamp_lower_idx)
    # feat 1: lower and upper -> sigmoid (clamp_lower_upper_idx)
    # feat 2: upper only -> softplus_upper (clamp_upper_idx)
    assert model.clamp_lower_idx.tolist() == [0]
    assert model.clamp_lower_upper_idx.tolist() == [1]
    assert model.clamp_upper_idx.tolist() == [2]

    # Verify get_clamped_new_state runs and produces finite outputs
    batch_size = 2
    num_nodes = dummy_datastore.num_grid_points
    num_vars = len(dummy_datastore.get_vars_names(category="state"))
    prev_state = torch.zeros(batch_size, num_nodes, num_vars)
    state_delta = torch.ones(batch_size, num_nodes, num_vars)

    new_state = model.get_clamped_new_state(state_delta, prev_state)
    assert new_state.shape == (batch_size, num_nodes, num_vars)
    assert torch.isfinite(new_state).all().item()
