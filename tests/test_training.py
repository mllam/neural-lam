# Standard library
import warnings
from pathlib import Path

# Third-party
import pytest
import pytorch_lightning as pl
import torch
import wandb

# First-party
from neural_lam import config as nlconfig
from neural_lam.create_graph_with_wmg import create_graph_from_datastore
from neural_lam.datastore import DATASTORES
from neural_lam.datastore.base import BaseRegularGridDatastore
from neural_lam.models import ARForecaster, ForecasterModule, GraphLAM
from neural_lam.weather_dataset import WeatherDataModule
from tests.conftest import init_datastore_example
from tests.dummy_datastore import DummyDatastore, set_framed_boundary

# Model architecture defaults for tests
GRAPH = "1level"
HIDDEN_DIM = 4
HIDDEN_LAYERS = 1
PROCESSOR_LAYERS = 2
MESH_AGGR = "sum"
NUM_PAST_FORCING_STEPS = 1
NUM_FUTURE_FORCING_STEPS = 1


def run_simple_training(
    datastore,
    set_output_std,
    metrics_watch=None,
    var_leads_metrics_watch=None,
):
    """
    Run one epoch of a simple model training setup using the given datastore.

    Parameters
    ----------
    datastore : BaseRegularGridDatastore
        Datastore to load data from for training
    set_output_std : bool
        If --output_std should be set during training
    """
    if metrics_watch is None:
        metrics_watch = []
    if var_leads_metrics_watch is None:
        var_leads_metrics_watch = {}

    if torch.cuda.is_available():
        device_name = "cuda"
        torch.set_float32_matmul_precision(
            "high"
        )  # Allows using Tensor Cores on A100s

        if torch.cuda.device_count() < 2:
            warnings.warn(
                "Running test suite on a single CUDA device. "
                "Multi-device testing still required.",
                UserWarning,
            )

    else:
        device_name = "cpu"

    trainer = pl.Trainer(
        max_epochs=1,
        deterministic=True,
        accelerator=device_name,
        # Dynamically allocate devices
        # to support single-GPU machines
        devices=2 if torch.cuda.device_count() >= 2 else 1,
        log_every_n_steps=1,
        # use `detect_anomaly` to ensure that we don't have NaNs popping up
        # during training
        detect_anomaly=True,
    )

    graph_name = "1level"

    graph_dir_path = Path(datastore.root_path) / "graph" / graph_name

    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            archetype="keisler",
        )

    data_module = WeatherDataModule(
        datastore=datastore,
        ar_steps_train=3,
        ar_steps_eval=5,
        batch_size=2,
        num_workers=1,
        num_past_forcing_steps=1,
        num_future_forcing_steps=1,
    )

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME, config_path=datastore.root_path
        )
    )

    # Build predictor and forecaster externally, then inject into
    # ForecasterModule
    # First-party
    from neural_lam.models import MODELS, ARForecaster

    predictor_class = MODELS["graph_lam"]
    predictor = predictor_class(
        datastore=datastore,
        graph_name=graph_name,
        hidden_dim=HIDDEN_DIM,
        hidden_layers=HIDDEN_LAYERS,
        processor_layers=PROCESSOR_LAYERS,
        mesh_aggr=MESH_AGGR,
        num_past_forcing_steps=NUM_PAST_FORCING_STEPS,
        num_future_forcing_steps=NUM_FUTURE_FORCING_STEPS,
        output_std=set_output_std,
        output_clamping_lower=config.training.output_clamping.lower,
        output_clamping_upper=config.training.output_clamping.upper,
    )
    forecaster = ARForecaster(predictor, datastore)

    model = ForecasterModule(
        forecaster=forecaster,
        config=config,
        datastore=datastore,
        loss="mse",
        lr=1.0e-3,
        restore_opt=False,
        n_example_pred=1,
        val_steps_to_log=[1, 3],
        metrics_watch=metrics_watch,
        var_leads_metrics_watch=var_leads_metrics_watch,
    )
    wandb.init(mode="disabled")  # Disable wandb for offline test run
    trainer.fit(model=model, datamodule=data_module)


@pytest.mark.slow
@pytest.mark.parametrize("datastore_name", DATASTORES.keys())
def test_training(datastore_name):
    datastore = init_datastore_example(datastore_name)

    if not isinstance(datastore, BaseRegularGridDatastore):
        pytest.skip(
            f"Skipping test for {datastore_name} as "
            f"it is not a regular grid datastore."
        )

    run_simple_training(datastore, set_output_std=False)


@pytest.mark.slow
def test_training_output_std():
    datastore = init_datastore_example("mdp")  # Test only with mdp datastore
    run_simple_training(datastore, set_output_std=True)


def test_all_gather_cat_single_device():
    """
    Test that all_gather_cat preserves tensor shape on single-device runs.
    On a single device, all_gather returns the tensor unchanged (no new
    leading dim), so all_gather_cat should not flatten any existing dims.
    """

    class MockModule:
        """Minimal object with mocked single-device all_gather."""

        def all_gather(self, tensor_to_gather, sync_grads=False):
            # Single-device behavior: return tensor unchanged
            return tensor_to_gather

    module = MockModule()
    get_fn = ForecasterModule.all_gather_cat.__get__
    module.all_gather_cat = get_fn(module, MockModule)  # type: ignore

    # Simulate a 3D metric tensor: (N_eval, pred_steps, d_f)
    tensor = torch.randn(4, 3, 5)
    result = module.all_gather_cat(tensor)  # type: ignore

    # On single device, shape must be preserved
    assert result.shape == tensor.shape, (
        f"all_gather_cat changed shape on single device: "
        f"{tensor.shape} -> {result.shape}"
    )
    assert torch.equal(result, tensor)


def test_all_gather_cat_multi_device_simulation():
    """
    Test that all_gather_cat correctly flattens when all_gather adds a
    leading dimension (simulating multi-device behavior).
    """

    class MockModule:
        """Object with mocked multi-device all_gather."""

        def all_gather(self, tensor, sync_grads=False):
            # Simulate 2-GPU all_gather: prepend a dim of size 2
            return torch.stack([tensor, tensor], dim=0)

    module = MockModule()
    get_fn = ForecasterModule.all_gather_cat.__get__
    module.all_gather_cat = get_fn(module, MockModule)  # type: ignore

    tensor = torch.randn(4, 3, 5)  # (N_eval, pred_steps, d_f)
    result = module.all_gather_cat(tensor)  # type: ignore

    # Should flatten (2, 4, 3, 5) -> (8, 3, 5)
    assert result.shape == (
        8,
        3,
        5,
    ), f"all_gather_cat wrong shape on multi-device: {result.shape}"
    # Validate values match expected concatenation along dim 0
    expected = torch.cat([tensor, tensor], dim=0)
    assert torch.equal(result, expected), (
        "all_gather_cat produced incorrectly ordered/combined values "
        "on multi-device simulation"
    )


@pytest.mark.slow
def test_test_step_excludes_boundary_from_spatial_loss(tmp_path):
    """
    Regression test for issue #569.

    `test_step` built its spatial loss map over every grid node, while every
    other loss and metric call in the module restricts itself to the interior
    via `interior_mask_bool`. Boundary nodes therefore leaked into the plotted
    loss maps and into the saved `mean_spatial_loss.pt`.

    Run `trainer.test()` end to end on a datastore with a known boundary mask
    and check that the saved map is NaN on the boundary and finite inside.
    """
    # 20x20 is the smallest grid `create_graph_from_datastore` still builds a
    # `1level` mesh for.
    datastore = DummyDatastore(n_grid_points=400, n_timesteps=10)
    set_framed_boundary(datastore)

    graph_dir_path = Path(datastore.root_path) / "graph" / GRAPH
    if not graph_dir_path.exists():
        create_graph_from_datastore(
            datastore=datastore,
            output_root_path=str(graph_dir_path),
            archetype="keisler",
        )

    config = nlconfig.NeuralLAMConfig(
        datastore=nlconfig.DatastoreSelection(
            kind=datastore.SHORT_NAME,
            config_path=datastore.root_path,
        ),
    )

    predictor = GraphLAM(
        datastore=datastore,
        graph_name=GRAPH,
        hidden_dim=HIDDEN_DIM,
        hidden_layers=HIDDEN_LAYERS,
        processor_layers=1,
        mesh_aggr=MESH_AGGR,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
        output_std=False,
        output_clamping_lower=config.training.output_clamping.lower,
        output_clamping_upper=config.training.output_clamping.upper,
    )
    model = ForecasterModule(
        forecaster=ARForecaster(predictor, datastore),
        config=config,
        datastore=datastore,
        loss="mse",
        lr=1.0e-3,
        restore_opt=False,
        n_example_pred=0,  # skip example plotting, not what is under test
        val_steps_to_log=[1],
    )

    data_module = WeatherDataModule(
        datastore=datastore,
        ar_steps_train=1,
        ar_steps_eval=2,
        batch_size=1,
        num_workers=0,
        num_past_forcing_steps=0,
        num_future_forcing_steps=0,
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=pl.loggers.CSVLogger(save_dir=str(tmp_path)),
        log_every_n_steps=1,
        enable_checkpointing=False,
    )
    trainer.test(model=model, datamodule=data_module)

    saved_path = Path(trainer.logger.save_dir) / "mean_spatial_loss.pt"
    assert saved_path.exists(), "test epoch did not save a spatial loss map"

    # (len(val_steps_to_log), num_grid_nodes)
    mean_spatial_loss = torch.load(saved_path, weights_only=True)
    assert mean_spatial_loss.shape == (1, datastore.num_grid_points)

    boundary = torch.tensor(datastore.boundary_mask.values, dtype=torch.bool)
    assert torch.isnan(mean_spatial_loss[:, boundary]).all(), (
        "boundary nodes must be excluded from the test spatial loss map, "
        "consistent with every other loss call in the module"
    )
    assert torch.isfinite(
        mean_spatial_loss[:, ~boundary]
    ).all(), "interior nodes must keep finite loss values"
