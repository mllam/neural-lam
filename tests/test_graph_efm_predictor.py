"""Unit tests for the Graph-EFM single-step probabilistic predictors.

These mirror the smoke-test pattern used for the deterministic predictors
(see ``tests/test_gnn_layers.py``): build the flat (GraphEFMMS) and
hierarchical (GraphEFM) variants on the real example datastore with a freshly
created graph, then exercise ``forward``, ``compute_step_loss`` and the
sampling helpers on synthetic tensors.
"""

# Standard library
from pathlib import Path

# Third-party
import pytest
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam import metrics
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.loss_weighting import get_state_feature_weighting
from neural_lam.models.step_predictors.graph.graph_efm import (
    GraphEFM,
    GraphEFMMS,
)
from tests.conftest import init_datastore_example

NUM_PAST_FORCING_STEPS = 1
NUM_FUTURE_FORCING_STEPS = 1


def _datastore_and_config_with_graph(graph_name):
    """Create the example datastore and ensure ``graph_name`` exists."""
    datastore = init_datastore_example("mdp")
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME,
            config_path=datastore.root_path,
        )
    )

    if graph_name == "hierarchical":
        hierarchical = True
        n_max_levels = 3
    else:
        hierarchical = False
        n_max_levels = 1

    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name
    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            hierarchical=hierarchical,
            n_max_levels=n_max_levels,
        )
    return datastore, config


def _build_predictor(graph_name, output_std=False):
    datastore, config = _datastore_and_config_with_graph(graph_name)
    if graph_name == "hierarchical":
        predictor_class = GraphEFM
        layer_kwargs = {
            "prior_intra_level_layers": 1,
            "encoder_intra_level_layers": 1,
            "decoder_intra_level_layers": 1,
        }
    else:
        predictor_class = GraphEFMMS
        layer_kwargs = {
            "prior_m2m_layers": 1,
            "encoder_m2m_layers": 1,
            "decoder_m2m_layers": 1,
        }
    predictor = predictor_class(
        config=config,
        datastore=datastore,
        graph_name=graph_name,
        hidden_dim=4,
        hidden_layers=1,
        latent_dim=4,
        learn_prior=True,
        prior_dist="isotropic",
        num_past_forcing_steps=NUM_PAST_FORCING_STEPS,
        num_future_forcing_steps=NUM_FUTURE_FORCING_STEPS,
        output_std=output_std,
        **layer_kwargs,
    )
    return predictor, datastore, config


def _make_inputs(predictor, datastore, batch_size=2):
    num_grid_nodes = predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing") * (
        NUM_PAST_FORCING_STEPS + NUM_FUTURE_FORCING_STEPS + 1
    )
    torch.manual_seed(0)
    prev_state = torch.randn(batch_size, num_grid_nodes, d_state)
    prev_prev_state = torch.randn(batch_size, num_grid_nodes, d_state)
    forcing = torch.randn(batch_size, num_grid_nodes, d_forcing)
    return prev_state, prev_prev_state, forcing, d_state


@pytest.mark.parametrize("graph_name", ["1level", "hierarchical"])
def test_forward_shapes_and_no_std(graph_name):
    """forward returns a (B, num_grid_nodes, d_state) state and None std when
    output_std is False, for both flat and hierarchical graphs."""
    predictor, datastore, _ = _build_predictor(graph_name)
    prev_state, prev_prev_state, forcing, d_state = _make_inputs(
        predictor, datastore
    )

    new_state, pred_std = predictor(prev_state, prev_prev_state, forcing)

    assert new_state.shape == (2, predictor.num_grid_nodes, d_state)
    assert pred_std is None


@pytest.mark.parametrize("graph_name", ["1level", "hierarchical"])
def test_forward_output_std_returns_std(graph_name):
    """With output_std=True the decoder produces a positive std of the same
    shape as the state."""
    predictor, datastore, _ = _build_predictor(graph_name, output_std=True)
    prev_state, prev_prev_state, forcing, d_state = _make_inputs(
        predictor, datastore
    )

    new_state, pred_std = predictor(prev_state, prev_prev_state, forcing)

    expected = (2, predictor.num_grid_nodes, d_state)
    assert new_state.shape == expected
    assert pred_std is not None
    assert pred_std.shape == expected
    assert (pred_std > 0).all()


@pytest.mark.parametrize("graph_name", ["1level", "hierarchical"])
def test_compute_step_loss_shapes_and_kl_toggle(graph_name):
    """compute_step_loss returns (likelihood (B,), kl, pred_mean, pred_std);
    kl is a (B,) tensor when compute_kl=True and None when disabled."""
    predictor, datastore, _ = _build_predictor(graph_name)
    prev_state, prev_prev_state, forcing, d_state = _make_inputs(
        predictor, datastore
    )
    B = prev_state.shape[0]
    prev_states = torch.stack([prev_prev_state, prev_state], dim=1)
    current_state = torch.randn(B, predictor.num_grid_nodes, d_state)
    interior_mask = torch.ones(predictor.num_grid_nodes, dtype=torch.bool)

    # KL on
    likelihood, kl, pred_mean, pred_std = predictor.compute_step_loss(
        prev_states,
        current_state,
        forcing,
        loss_fn=metrics.nll,
        interior_mask=interior_mask,
        compute_kl=True,
    )
    assert likelihood.shape == (B,)
    assert kl is not None
    assert kl.shape == (B,)
    assert pred_mean.shape == (B, predictor.num_grid_nodes, d_state)
    # output_std=False -> constant per-variable std (d_state,)
    assert pred_std.shape == (d_state,)

    # KL off -> kl_term is None
    likelihood_off, kl_off, _, _ = predictor.compute_step_loss(
        prev_states,
        current_state,
        forcing,
        loss_fn=metrics.nll,
        interior_mask=interior_mask,
        compute_kl=False,
    )
    assert kl_off is None
    assert likelihood_off.shape == (B,)


@pytest.mark.parametrize("graph_name", ["1level", "hierarchical"])
def test_compute_step_loss_is_differentiable(graph_name):
    """The ELBO pieces are differentiable through the rsample paths, and the
    gradient reaches encoder, decoder and prior parameters."""
    predictor, datastore, _ = _build_predictor(graph_name)
    prev_state, prev_prev_state, forcing, d_state = _make_inputs(
        predictor, datastore
    )
    B = prev_state.shape[0]
    prev_states = torch.stack([prev_prev_state, prev_state], dim=1)
    current_state = torch.randn(B, predictor.num_grid_nodes, d_state)
    interior_mask = torch.ones(predictor.num_grid_nodes, dtype=torch.bool)

    likelihood, kl, _, _ = predictor.compute_step_loss(
        prev_states,
        current_state,
        forcing,
        loss_fn=metrics.nll,
        interior_mask=interior_mask,
        compute_kl=True,
    )
    elbo = (likelihood - kl).mean()
    elbo.backward()

    for module in (predictor.encoder, predictor.decoder, predictor.prior_model):
        assert any(
            p.grad is not None and torch.any(p.grad != 0)
            for p in module.parameters()
        ), f"no gradient reached {module.__class__.__name__}"


@pytest.mark.parametrize("graph_name", ["1level", "hierarchical"])
def test_forward_member_stochasticity(graph_name):
    """Two forward calls with identical inputs differ, because the latent is
    resampled from the prior each call (catches an unused-latent regression)."""
    predictor, datastore, _ = _build_predictor(graph_name)
    prev_state, prev_prev_state, forcing, _ = _make_inputs(predictor, datastore)

    out_a, _ = predictor(prev_state, prev_prev_state, forcing)
    out_b, _ = predictor(prev_state, prev_prev_state, forcing)

    assert not torch.allclose(out_a, out_b)


def test_per_var_std_matches_module_formula():
    """per_var_std mirrors ForecasterModule's formula:
    state_diff_std_standardized / sqrt(state_feature_weights)."""
    predictor, datastore, config = _build_predictor("1level")

    da_state_stats = datastore.get_standardization_dataarray(category="state")
    diff_std = torch.tensor(
        da_state_stats.state_diff_std_standardized.values,
        dtype=torch.float32,
    )
    feature_weights = torch.tensor(
        get_state_feature_weighting(config=config, datastore=datastore),
        dtype=torch.float32,
    )
    expected = diff_std / torch.sqrt(feature_weights)

    assert predictor.per_var_std is not None
    assert torch.allclose(predictor.per_var_std, expected)


def test_per_var_std_none_when_output_std():
    """When the decoder outputs its own std, the constant per_var_std is unused
    and left as None (mirrors ForecasterModule)."""
    predictor, _, _ = _build_predictor("1level", output_std=True)
    assert predictor.per_var_std is None


@pytest.mark.parametrize(
    "predictor_class, graph_name",
    [(GraphEFM, "1level"), (GraphEFMMS, "hierarchical")],
)
def test_graph_type_mismatch_raises(predictor_class, graph_name):
    """GraphEFM requires a hierarchical graph and GraphEFMMS a flat one;
    constructing with the wrong graph type raises ValueError."""
    datastore, config = _datastore_and_config_with_graph(graph_name)
    with pytest.raises(ValueError, match="mesh graph"):
        predictor_class(
            config=config,
            datastore=datastore,
            graph_name=graph_name,
            hidden_dim=4,
            hidden_layers=1,
        )
