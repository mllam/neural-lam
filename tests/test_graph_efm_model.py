"""Integration tests for the full Graph-EFM model.

Exercises the pieces that turn the Graph-EFM single-step predictors into a
trainable model: the ``GraphEFMForecaster`` ELBO objective, its inherited
ensemble sampling, the ``ProbabilisticForecasterModule`` wrapping, and the
config-aware assembly path in ``train_model`` (``build_predictor`` /
``build_forecaster_module``). Predictors are built on the real example
datastore with a freshly created graph, mirroring
``tests/test_graph_efm_predictor.py``.
"""

# Standard library
from argparse import Namespace
from pathlib import Path

# Third-party
import pytest
import torch

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph import create_graph_from_datastore
from neural_lam.models import (
    GraphEFM,
    GraphEFMForecaster,
    GraphEFMMultiScale,
    ProbabilisticForecasterModule,
)
from neural_lam.train_model import build_forecaster_module, build_predictor
from tests.conftest import init_datastore_example

NUM_PAST_FORCING_STEPS = 1
NUM_FUTURE_FORCING_STEPS = 1


def _datastore_and_config(graph_name):
    """
    Build the example datastore + config and ensure ``graph_name`` exists.

    Parameters
    ----------
    graph_name : str
        Graph directory name; ``"hierarchical"`` builds a multi-level graph,
        anything else a flat one.

    Returns
    -------
    datastore : BaseDatastore
        The example ``mdp`` datastore.
    config : NeuralLAMConfig
        A configuration selecting that datastore.
    """
    datastore = init_datastore_example("mdp")
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )

    hierarchical = graph_name == "hierarchical"
    n_max_levels = 3 if hierarchical else 1
    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name
    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            hierarchical=hierarchical,
            n_max_levels=n_max_levels,
        )
    return datastore, config


def _build_predictor(datastore, graph_name, output_std=False):
    """
    Construct a small Graph-EFM predictor for ``graph_name``.

    Parameters
    ----------
    datastore : BaseDatastore
        Datastore to build the predictor on.
    graph_name : str
        Graph directory name selecting the flat vs hierarchical variant.
    output_std : bool, default False
        Whether the decoder outputs its own std.

    Returns
    -------
    BaseGraphEFM
        The constructed predictor.
    """
    if graph_name == "hierarchical":
        predictor_class = GraphEFM
        layer_kwargs = {
            "prior_intra_level_layers": 1,
            "encoder_intra_level_layers": 1,
            "decoder_intra_level_layers": 1,
        }
    else:
        predictor_class = GraphEFMMultiScale
        layer_kwargs = {
            "prior_m2m_layers": 1,
            "encoder_m2m_layers": 1,
            "decoder_m2m_layers": 1,
        }
    return predictor_class(
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


def _example_batch(datastore, predictor, batch_size=2, pred_steps=3):
    """
    Build a synthetic ``(init_states, forcing_features, target_states)`` batch.

    Parameters
    ----------
    datastore : BaseDatastore
        Datastore providing variable counts.
    predictor : BaseGraphEFM
        Predictor providing the grid node count.
    batch_size : int, default 2
        Number of samples in the batch.
    pred_steps : int, default 3
        Rollout length.

    Returns
    -------
    init_states : torch.Tensor
        Shape ``(B, 2, num_grid_nodes, d_state)``.
    forcing_features : torch.Tensor
        Shape ``(B, pred_steps, num_grid_nodes, d_forcing)``.
    target_states : torch.Tensor
        Shape ``(B, pred_steps, num_grid_nodes, d_state)``.
    """
    num_grid_nodes = predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing") * (
        NUM_PAST_FORCING_STEPS + NUM_FUTURE_FORCING_STEPS + 1
    )
    torch.manual_seed(0)
    init_states = torch.randn(batch_size, 2, num_grid_nodes, d_state)
    forcing_features = torch.randn(
        batch_size, pred_steps, num_grid_nodes, d_forcing
    )
    target_states = torch.randn(batch_size, pred_steps, num_grid_nodes, d_state)
    return init_states, forcing_features, target_states


@pytest.mark.parametrize("graph_name", ["1level", "hierarchical"])
def test_forecaster_forward_and_ensemble_shapes(graph_name):
    """The forecaster unrolls a prior-sampled rollout and stacks members into
    an ensemble of the documented shape."""
    datastore, config = _datastore_and_config(graph_name)
    predictor = _build_predictor(datastore, graph_name)
    forecaster = GraphEFMForecaster(predictor, datastore, config=config)

    B, pred_steps, num_members = 2, 3, 4
    init_states, forcing_features, target_states = _example_batch(
        datastore, predictor, batch_size=B, pred_steps=pred_steps
    )
    d_state = target_states.shape[-1]
    num_grid_nodes = predictor.num_grid_nodes

    prediction, pred_std = forecaster(
        init_states, forcing_features, target_states
    )
    assert prediction.shape == (B, pred_steps, num_grid_nodes, d_state)
    assert pred_std is None  # output_std=False predictor

    ensemble, per_member_std = forecaster.sample_ensemble(
        init_states, forcing_features, target_states, num_members=num_members
    )
    assert ensemble.shape == (
        B,
        num_members,
        pred_steps,
        num_grid_nodes,
        d_state,
    )
    assert per_member_std is None
    # Members carry independent latent samples
    assert not torch.allclose(ensemble[:, 0], ensemble[:, 1])


@pytest.mark.parametrize("graph_name", ["1level", "hierarchical"])
def test_predictor_step_distributions_contract(graph_name):
    """step_distributions reuses the shared graph embedding and returns the
    prior on the inference path, and the posterior (with the prior gated on
    compute_prior) on the training path."""
    datastore, _ = _datastore_and_config(graph_name)
    predictor = _build_predictor(datastore, graph_name)
    init_states, forcing_features, target_states = _example_batch(
        datastore, predictor, batch_size=2, pred_steps=1
    )
    prev_prev_state, prev_state = init_states[:, 0], init_states[:, 1]
    forcing, target_state = forcing_features[:, 0], target_states[:, 0]
    d_state = target_state.shape[-1]
    num_grid_nodes = predictor.num_grid_nodes

    graph_emb = predictor.embedd_graph(2)
    assert {"g2m", "m2g", "mesh"} <= set(graph_emb)

    # Inference path: latent from the prior, no posterior
    prior, posterior, pred_mean, pred_std = predictor.step_distributions(
        prev_state, prev_prev_state, forcing, graph_emb, target_state=None
    )
    assert prior is not None and posterior is None
    assert pred_mean.shape == (2, num_grid_nodes, d_state)
    assert pred_std is None

    # Training path with KL: both distributions present
    prior, posterior, _, _ = predictor.step_distributions(
        prev_state,
        prev_prev_state,
        forcing,
        graph_emb,
        target_state=target_state,
        compute_prior=True,
    )
    assert prior is not None and posterior is not None

    # Training path without KL: prior skipped
    prior, posterior, _, _ = predictor.step_distributions(
        prev_state,
        prev_prev_state,
        forcing,
        graph_emb,
        target_state=target_state,
        compute_prior=False,
    )
    assert prior is None and posterior is not None


def test_predictor_clamps_predicted_mean():
    """With output clamping configured for a feature, Graph-EFM keeps the
    predicted mean for that feature within the configured bounds, clamping it
    like the deterministic models do."""
    datastore, _ = _datastore_and_config("1level")
    state_names = datastore.get_vars_names(category="state")
    lower, upper = -0.5, 0.5
    predictor = GraphEFMMultiScale(
        datastore=datastore,
        graph_name="1level",
        hidden_dim=4,
        hidden_layers=1,
        latent_dim=4,
        prior_m2m_layers=1,
        encoder_m2m_layers=1,
        decoder_m2m_layers=1,
        output_clamping_lower={state_names[0]: lower},
        output_clamping_upper={state_names[0]: upper},
    )
    # The first state feature has a two-sided (sigmoid) clamp registered
    assert predictor.clamp_lower_upper_idx.tolist() == [0]
    lower_n = (lower - predictor.state_mean[0]) / predictor.state_std[0]
    upper_n = (upper - predictor.state_mean[0]) / predictor.state_std[0]

    B = 2
    num_grid_nodes = predictor.num_grid_nodes
    d_state = len(state_names)
    d_forcing = datastore.get_num_data_vars(category="forcing") * (
        NUM_PAST_FORCING_STEPS + NUM_FUTURE_FORCING_STEPS + 1
    )
    torch.manual_seed(0)
    prev_state = torch.randn(B, num_grid_nodes, d_state)
    # The current value of the clamped feature must be within its bounds so
    # the inverse clamp is finite; the midpoint is a safe choice.
    prev_state[..., 0] = (lower_n + upper_n) / 2
    prev_prev_state = torch.randn(B, num_grid_nodes, d_state)
    forcing = torch.randn(B, num_grid_nodes, d_forcing)

    pred_mean, _ = predictor(prev_state, prev_prev_state, forcing)

    clamped_feature = pred_mean[..., 0]
    assert torch.all(clamped_feature > lower_n)
    assert torch.all(clamped_feature < upper_n)


def test_hierarchical_zero_intra_level_layers_runs():
    """A hierarchical GraphEFM with no intra-level layers uses an empty m2m
    placeholder; forward must still run (regression for m2m handling)."""
    datastore, _ = _datastore_and_config("hierarchical")
    predictor = GraphEFM(
        datastore=datastore,
        graph_name="hierarchical",
        hidden_dim=4,
        hidden_layers=1,
        latent_dim=4,
        prior_intra_level_layers=0,
        encoder_intra_level_layers=0,
        decoder_intra_level_layers=0,
    )
    assert not predictor.embedd_m2m

    B = 2
    num_grid_nodes = predictor.num_grid_nodes
    d_state = datastore.get_num_data_vars(category="state")
    d_forcing = datastore.get_num_data_vars(category="forcing") * (
        NUM_PAST_FORCING_STEPS + NUM_FUTURE_FORCING_STEPS + 1
    )
    torch.manual_seed(0)
    prev_state = torch.randn(B, num_grid_nodes, d_state)
    prev_prev_state = torch.randn(B, num_grid_nodes, d_state)
    forcing = torch.randn(B, num_grid_nodes, d_forcing)

    pred_mean, _ = predictor(prev_state, prev_prev_state, forcing)
    assert pred_mean.shape == (B, num_grid_nodes, d_state)


@pytest.mark.parametrize("graph_name", ["1level", "hierarchical"])
def test_elbo_training_loss_gradient_flow(graph_name):
    """compute_training_loss returns a finite scalar ELBO with likelihood/KL
    components, and gradients flow back into the predictor."""
    datastore, config = _datastore_and_config(graph_name)
    predictor = _build_predictor(datastore, graph_name)
    forecaster = GraphEFMForecaster(
        predictor, datastore, config=config, loss="mse", kl_beta=1.0
    )

    init_states, forcing_features, target_states = _example_batch(
        datastore, predictor
    )
    interior_mask_bool = forecaster.interior_mask[0, :, 0].to(torch.bool)

    torch.manual_seed(0)
    batch_loss, loss_components = forecaster.compute_training_loss(
        init_states,
        forcing_features,
        target_states,
        interior_mask_bool=interior_mask_bool,
    )

    assert batch_loss.shape == ()
    assert torch.isfinite(batch_loss)
    assert set(loss_components) == {"elbo_likelihood", "elbo_kl", "elbo"}
    assert (loss_components["elbo_kl"] >= 0).all()

    batch_loss.backward()
    grads = [p.grad for p in predictor.parameters() if p.grad is not None]
    assert grads, "no gradients reached the predictor"
    assert any(torch.any(g != 0) for g in grads)


@pytest.mark.parametrize("graph_name", ["1level", "hierarchical"])
def test_elbo_kl_beta_zero_skips_kl(graph_name):
    """With kl_beta=0 the loss is the negative likelihood alone and no KL
    component is reported (pure auto-encoder training)."""
    datastore, config = _datastore_and_config(graph_name)
    predictor = _build_predictor(datastore, graph_name)
    forecaster = GraphEFMForecaster(
        predictor, datastore, config=config, loss="mse", kl_beta=0.0
    )

    init_states, forcing_features, target_states = _example_batch(
        datastore, predictor
    )
    interior_mask_bool = forecaster.interior_mask[0, :, 0].to(torch.bool)

    torch.manual_seed(0)
    batch_loss, loss_components = forecaster.compute_training_loss(
        init_states,
        forcing_features,
        target_states,
        interior_mask_bool=interior_mask_bool,
    )

    assert set(loss_components) == {"elbo_likelihood"}
    torch.testing.assert_close(batch_loss, -loss_components["elbo_likelihood"])
    batch_loss.backward()  # still differentiable


def test_module_training_and_validation_steps():
    """The ProbabilisticForecasterModule delegates training to the forecaster
    ELBO and scores an ensemble mean during validation."""
    datastore, config = _datastore_and_config("1level")
    predictor = _build_predictor(datastore, "1level")
    forecaster = GraphEFMForecaster(
        predictor, datastore, config=config, loss="mse", kl_beta=1.0
    )
    model = ProbabilisticForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        eval_ensemble_size=2,
    )

    B, pred_steps = 2, 3
    init_states, forcing_features, target_states = _example_batch(
        datastore, predictor, batch_size=B, pred_steps=pred_steps
    )
    batch_times = torch.zeros(B, pred_steps)
    batch = (init_states, target_states, forcing_features, batch_times)

    torch.manual_seed(0)
    train_loss = model.training_step(batch)
    assert train_loss.shape == ()
    assert torch.isfinite(train_loss)

    model.validation_step(batch, 0)
    (entry_mses,) = model.val_metrics["ens_mse"]
    d_state = target_states.shape[-1]
    assert entry_mses.shape == (B, pred_steps, d_state)
    assert torch.all(torch.isfinite(entry_mses))


@pytest.mark.parametrize(
    "model_name, predictor_class, graph_name",
    [
        ("graph_efm", GraphEFM, "hierarchical"),
        ("graph_efm_ms", GraphEFMMultiScale, "1level"),
    ],
)
def test_train_model_assembly_selects_probabilistic_path(
    model_name, predictor_class, graph_name
):
    """train_model's build_predictor/build_forecaster_module route the
    graph_efm* models through the probabilistic assembly, reading the
    Graph-EFM hyperparameters from the ``probabilistic`` config section and
    producing the right predictor, forecaster, module class and checkpoint
    monitor."""
    datastore, _ = _datastore_and_config(graph_name)
    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        ),
        probabilistic=nlconfig.ProbabilisticConfig(
            latent_dim=4,
            prior_layers=1,
            encoder_layers=1,
            decoder_layers=1,
            kl_beta=0.5,
            eval_ensemble_size=3,
        ),
    )
    args = Namespace(
        model=model_name,
        graph=graph_name,
        hidden_dim=4,
        hidden_layers=1,
        num_past_forcing_steps=NUM_PAST_FORCING_STEPS,
        num_future_forcing_steps=NUM_FUTURE_FORCING_STEPS,
        output_std=False,
        g2m_gnn_type="InteractionNet",
        m2g_gnn_type="InteractionNet",
        loss="mse",
    )

    predictor = build_predictor(args, config, datastore)
    assert isinstance(predictor, predictor_class)
    # Hyperparameters came from config.probabilistic, not the CLI args
    assert predictor.latent_dim == 4

    forecaster, module_class, module_kwargs, val_monitor = (
        build_forecaster_module(args, config, datastore, predictor)
    )
    assert isinstance(forecaster, GraphEFMForecaster)
    assert forecaster.kl_beta == 0.5
    assert module_class is ProbabilisticForecasterModule
    assert module_kwargs == {"eval_ensemble_size": 3}
    assert val_monitor == "val_mean_ens_rmse"
